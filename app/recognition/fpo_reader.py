"""OCR pronto: fast-plate-ocr (MIT, ONNX Runtime, CPU). Implementa o Protocol `PlateReader`."""

import cv2
import numpy as np

from app.recognition.base import PlateReading


def limpar(texto: str) -> str:
    """Maiúsculas, sem hífen/espaço/padding ('_'). Texto cru: não corrige caracteres."""
    return "".join(c for c in texto.upper() if c.isalnum())


class FastPlateReader:
    """Leitor de placa sobre um `LicensePlateRecognizer` já carregado (injetável em testes).

    Confiança = média das probabilidades por caractere (`char_probs`) apenas das posições
    com probabilidade > 0 (as posições de padding vêm zeradas), 0.0 se nenhuma.
    """

    def __init__(self, recognizer, model_name: str = "") -> None:
        self._rec = recognizer
        self.model_name = model_name
        self._cinza = "gray" in str(recognizer.config.image_color_mode).lower()

    @classmethod
    def from_model(cls, name: str) -> "FastPlateReader":
        # Importação tardia: só o modo fpo paga o custo. Baixa o modelo na 1ª vez (precisa de
        # internet); depois usa o cache local (~/.cache/fast-plate-ocr).
        from fast_plate_ocr import LicensePlateRecognizer

        return cls(LicensePlateRecognizer(name, device="cpu"), name)

    def read(self, crop: np.ndarray) -> PlateReading | None:
        if crop.ndim != 3 or crop.shape[0] < 2 or crop.shape[1] < 2:
            return None
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY if self._cinza else cv2.COLOR_BGR2RGB)
        pred = self._rec.run_one(img, return_confidence=True)
        texto = limpar(pred.plate)
        if not texto:
            return None
        probs = pred.char_probs
        conf = float(np.mean(probs[probs > 0])) if probs is not None and np.any(probs > 0) else 0.0
        return PlateReading(text=texto, confidence=conf)
