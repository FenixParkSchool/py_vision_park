import logging
import time
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone

import numpy as np

from app.recognition.base import PlateDetector, PlateReader
from app.recognition.preprocessing import crop_region
from app.schemas.recognition import PlateObservation
from app.services.deduplicator import Deduplicator
from app.services.voter import PlateVoter

log = logging.getLogger(__name__)


class RecognitionPipeline:
    """Leva um frame por: detecção -> recorte -> leitura -> observação."""

    def __init__(
        self,
        detector: PlateDetector,
        reader: PlateReader,
        camera_id: str,
        min_confidence: float,
        deduplicator: Deduplicator | None = None,
        ocr_min_confidence: float | None = None,
        voter: PlateVoter | None = None,
    ) -> None:
        self._detector = detector
        self._reader = reader
        self._camera_id = camera_id
        # Vale para a detecção e para a leitura, cada uma comparada com este valor.
        # Valor provisório de desenvolvimento, ainda não validado com dados reais.
        self._min_confidence = min_confidence
        self._ocr_min_confidence = min_confidence if ocr_min_confidence is None else ocr_min_confidence
        self._deduplicator = deduplicator
        self._voter = voter

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
            if leitura.confidence < self._ocr_min_confidence:
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
        """Ordem: leituras do quadro -> votação entre quadros (se houver) -> deduplicação.

        A votação junta as leituras de um mesmo veículo e solta uma só; o Deduplicator
        continua barrando o mesmo veículo que reaparece dentro da janela.
        """
        for frame in frames:
            # Instante em que o frame chegou ao pipeline, antes da inferência demorada.
            captured_at = datetime.now(timezone.utc)
            observacoes = self.process_frame(frame, captured_at)
            if self._voter is not None:
                # chamado em TODO quadro, mesmo sem leitura: é assim que o grupo percebe
                # que o veículo saiu
                observacoes = self._voter.push_frame(observacoes)
            yield from self._sem_repetidas(observacoes)
        if self._voter is not None:
            # fim do vídeo ou câmera parada: não perder o veículo que ainda estava em votação
            yield from self._sem_repetidas(self._voter.flush())

    def _sem_repetidas(self, observacoes: list[PlateObservation]) -> Iterator[PlateObservation]:
        for observacao in observacoes:
            if self._deduplicator is not None and self._deduplicator.is_duplicate(
                observacao.camera_id, observacao.plate_text
            ):
                log.debug("Leitura repetida suprimida")
                continue
            yield observacao