import logging
import threading
from collections.abc import Iterator

import numpy as np
from pydantic import ValidationError

from app.recognition.plate_format import normalize_plate
from app.schemas.events import PlateEvent
from app.services.backend_client import BackendClient, BackendRejected, BackendUnavailable

log = logging.getLogger(__name__)


class AlprWorker:
    """Roda o loop camera -> pipeline -> backend em uma thread própria (nunca em rota HTTP).

    Política: falha de envio ao backend (rejeitado/indisponível) perde o evento, é logada
    e o worker continua. Erro de câmera, detector ou leitor para o worker (estado failed).
    """

    def __init__(self, camera, pipeline, client: BackendClient, join_timeout_s: float = 5.0) -> None:
        self._camera = camera
        self._pipeline = pipeline
        self._client = client
        self._join_timeout_s = join_timeout_s
        self._thread: threading.Thread | None = None
        self._camera_open = False
        self._last_error: str | None = None
        self.frames = 0
        self.events_sent = 0
        self.duplicates = 0
        self.send_failures = 0
        self.invalid_plates = 0

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def ready(self) -> bool:
        return self.running and self._camera_open and self._last_error is None

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Worker já iniciado")
        self._thread = threading.Thread(target=self._run, name="alpr-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._camera.stop()
        if self._thread is not None:
            self._thread.join(timeout=self._join_timeout_s)
            if self._thread.is_alive():
                log.warning("Worker não encerrou dentro do timeout")
        self._camera.close()  # uma câmera USB só abre uma vez: sempre liberar
        self._camera_open = False

    def _counted_frames(self) -> Iterator[np.ndarray]:
        for frame in self._camera.frames():
            self.frames += 1
            yield frame

    def _run(self) -> None:
        try:
            self._camera.open()
            self._camera_open = True
            for obs in self._pipeline.process_stream(self._counted_frames()):
                self._send(obs)
        except Exception as exc:  # câmera, detector ou leitor: o worker para
            self._last_error = f"{type(exc).__name__}: {exc}"
            log.error("Worker parou: %s", self._last_error)
        finally:
            self._camera_open = False

    def _send(self, obs) -> None:
        # Só placas Mercosul/antigas seguem para o backend; o texto cru não é logado por padrão.
        plate = normalize_plate(obs.plate_text)
        if not plate.is_valid:
            self.invalid_plates += 1
            log.info("Leitura descartada: formato de placa inválido")
            return
        try:
            event = PlateEvent.from_observation(obs)
            if self._client.send(event):
                self.duplicates += 1
            else:
                self.events_sent += 1
        except (BackendRejected, BackendUnavailable) as exc:
            self.send_failures += 1
            log.warning("Evento perdido (%s): %s", type(exc).__name__, str(exc)[:100])
        except ValidationError:
            self.send_failures += 1
            log.warning("Evento descartado: placa inválida após normalização")
