import logging
from contextlib import closing
from pathlib import Path

import cv2

from app.camera.source import CameraSource
from app.config import get_settings
from app.recognition.base import PlateDetection, PlateReading
from app.recognition.fakes import FakePlateDetector
from app.schemas.recognition import BoundingBox
from app.services.pipeline import RecognitionPipeline

logging.basicConfig(level=logging.INFO)

settings = get_settings()
saida = Path("experiments/crops")
saida.mkdir(parents=True, exist_ok=True)

# CHUTE INICIAL: ajuste olhando um frame de experiments/frames/
CAIXA = BoundingBox(x=900, y=910, width=170, height=85)


class LeitorQueSalvaRecortes:
    """Leitor de teste: grava cada recorte que recebe e devolve sempre a mesma leitura."""

    def __init__(self) -> None:
        self.contador = 0

    def read(self, crop):
        cv2.imwrite(str(saida / f"recorte_{self.contador:04d}.png"), crop)
        self.contador += 1
        return PlateReading(text="ABC1D23", confidence=0.9)


pipeline = RecognitionPipeline(
    detector=FakePlateDetector([PlateDetection(bbox=CAIXA, confidence=0.9)]),
    reader=LeitorQueSalvaRecortes(),
    camera_id=settings.camera_id,
    min_confidence=settings.min_confidence,
)

camera = CameraSource(settings.camera_source, settings.sample_interval_seconds)

total = 0
with closing(camera.frames()) as frames:
    for observacao in pipeline.process_stream(frames):
        total += 1
        if total == 1:
            print("Primeira observação:", observacao)

print("Observações geradas:", total)