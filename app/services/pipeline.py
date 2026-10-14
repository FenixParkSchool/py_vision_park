import logging
import time
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone

import numpy as np

from app.recognition.base import PlateDetector, PlateReader
from app.recognition.preprocessing import crop_region
from app.schemas.recognition import PlateObservation

log = logging.getLogger(__name__)


class RecognitionPipeline:
    """Leva um frame por: detecção -> recorte -> leitura -> observação."""

    def __init__(
        self,
        detector: PlateDetector,
        reader: PlateReader,
        camera_id: str,
        min_confidence: float,
    ) -> None:
        self._detector = detector
        self._reader = reader
        self._camera_id = camera_id
        # Vale para a detecção e para a leitura, cada uma comparada com este valor.
        # Valor provisório de desenvolvimento, ainda não validado com dados reais.
        self._min_confidence = min_confidence

    def process_frame(self, frame: np.ndarray, captured_at: datetime) -> list[PlateObservation]:
        inicio = time.perf_counter()
        observacoes: list[PlateObservation] = []

        deteccoes = sorted(self._detector.detect(frame), key=lambda d: d.confidence, reverse=True)
        for deteccao in deteccoes:
            if deteccao.confidence < self._min_confidence:
                continue
            recorte = crop_region(frame, deteccao.bbox)
            if recorte is None:
                log.debug("Caixa fora do frame; detecção descartada")
                continue
            leitura = self._reader.read(recorte)
            if leitura is None or not leitura.text.strip():
                log.debug("Placa detectada, mas ilegível")
                continue
            if leitura.confidence < self._min_confidence:
                continue
            observacoes.append(
                PlateObservation(
                    camera_id=self._camera_id,
                    plate_text=leitura.text,
                    detection_confidence=deteccao.confidence,
                    ocr_confidence=leitura.confidence,
                    captured_at=captured_at,
                    bbox=deteccao.bbox,
                )
            )

        log.debug(
            "Frame processado em %.1f ms, %d leitura(s)",
            (time.perf_counter() - inicio) * 1000,
            len(observacoes),
        )
        return observacoes

    def process_stream(self, frames: Iterable[np.ndarray]) -> Iterator[PlateObservation]:
        for frame in frames:
            # Instante em que o frame chegou ao pipeline, antes da inferência demorada.
            captured_at = datetime.now(timezone.utc)
            yield from self.process_frame(frame, captured_at)