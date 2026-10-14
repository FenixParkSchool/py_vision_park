from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.schemas.recognition import BoundingBox


@dataclass(frozen=True)
class PlateDetection:
    """Uma região do frame onde o detector acredita haver uma placa."""

    bbox: BoundingBox
    confidence: float


@dataclass(frozen=True)
class PlateReading:
    """Texto lido pelo OCR em um recorte de placa."""

    text: str
    confidence: float


class PlateDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[PlateDetection]: ...


class PlateReader(Protocol):
    def read(self, crop: np.ndarray) -> PlateReading | None: ...