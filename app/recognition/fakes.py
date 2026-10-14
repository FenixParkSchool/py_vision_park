import numpy as np

from app.recognition.base import PlateDetection, PlateReading


class FakePlateDetector:
    """Detector falso: sempre devolve as mesmas detecções, qualquer que seja o frame."""

    def __init__(self, detections: list[PlateDetection] | None = None) -> None:
        self._detections = list(detections or [])

    def detect(self, frame: np.ndarray) -> list[PlateDetection]:
        return list(self._detections)


class FakePlateReader:
    """Leitor falso: sempre devolve a mesma leitura e guarda os recortes que recebeu."""

    def __init__(self, reading: PlateReading | None) -> None:
        self._reading = reading
        self.crops: list[np.ndarray] = []

    def read(self, crop: np.ndarray) -> PlateReading | None:
        self.crops.append(crop)
        return self._reading