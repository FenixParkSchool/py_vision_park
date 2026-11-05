"""Detector e leitor (OCR) reais, com modelos Keras treinados no RodoSol-ALPR.

Contrato de entrada/saída (vale para treino e inferência; o treino importa estas funções):
- Frame e recorte chegam em BGR uint8 (padrão do OpenCV). Os dois modelos usam só
  tons de cinza (cv2.COLOR_BGR2GRAY), então a ordem de cor não pesa no modelo.
- Normalização: pixel / 255 -> float32 em [0, 1], formato (lote, altura, largura, 1).
- Detector: rede totalmente convolucional que devolve um mapa de probabilidade
  "isto é placa" com 1/4 da resolução da entrada. Caixas saem dos componentes conexos.
- OCR: 7 classificadores (um por posição) sobre o alfabeto 0-9A-Z.

TensorFlow só é importado em `from_file`, nunca na importação deste módulo.
"""

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from app.recognition.base import PlateDetection, PlateReading
from app.recognition.preprocessing import guess_vehicle_type, prepare_plate
from app.schemas.recognition import BoundingBox

DETECTOR_PATH = Path("models/detector.keras")
OCR_PATH = Path("models/ocr.keras")

# Largura para a qual o frame é redimensionado antes do detector. No treino,
# os frames 1280x720 do RodoSol viram 384x216 (escala 0,3) e recortes ampliados até 2x.
DETECTOR_INPUT_WIDTH = 384
DETECTOR_STRIDE = 4  # o mapa de saída tem 1/4 da largura e da altura da entrada
DETECTOR_MULTIPLE = 8  # entrada é completada com zeros até múltiplo de 8 (2 reduções + 1 ampliação)

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PLATE_LENGTH = 7

Predict = Callable[[np.ndarray], np.ndarray]


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def detector_input(frame: np.ndarray, input_width: int = DETECTOR_INPUT_WIDTH) -> tuple[np.ndarray, float]:
    """Frame BGR -> tensor (1, H, W, 1) em [0, 1] e a escala usada (entrada / frame)."""
    cinza = to_gray(frame)
    altura, largura = cinza.shape
    escala = input_width / largura
    nova_altura = max(1, round(altura * escala))
    reduzido = cv2.resize(cinza, (input_width, nova_altura), interpolation=cv2.INTER_AREA)
    pad_h = -nova_altura % DETECTOR_MULTIPLE
    pad_w = -input_width % DETECTOR_MULTIPLE
    reduzido = np.pad(reduzido, ((0, pad_h), (0, pad_w)))
    return (reduzido.astype(np.float32) / 255.0)[None, :, :, None], escala


def mask_to_detections(
    prob: np.ndarray,
    scale: float,
    frame_shape: tuple[int, ...],
    threshold: float = 0.5,
    min_cells: int = 2,
) -> list[PlateDetection]:
    """Mapa de probabilidade (h, w) -> caixas no frame original.

    Cada grupo conexo de células acima do limiar vira uma caixa; a confiança é a
    probabilidade média do grupo.
    """
    binario = (prob >= threshold).astype(np.uint8)
    n, rotulos, stats, _ = cv2.connectedComponentsWithStats(binario, connectivity=8)
    altura, largura = frame_shape[:2]
    fator = DETECTOR_STRIDE / scale  # célula do mapa -> pixels do frame
    deteccoes: list[PlateDetection] = []
    for i in range(1, n):  # 0 é o fundo
        cx, cy, cw, ch, area = stats[i]
        if area < min_cells:
            continue
        x1 = max(0, int(cx * fator))
        y1 = max(0, int(cy * fator))
        x2 = min(largura, int(np.ceil((cx + cw) * fator)))
        y2 = min(altura, int(np.ceil((cy + ch) * fator)))
        if x2 <= x1 or y2 <= y1:
            continue
        confianca = float(prob[rotulos == i].mean())
        deteccoes.append(
            PlateDetection(BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1), confianca)
        )
    return deteccoes


def ocr_input(crop: np.ndarray) -> np.ndarray:
    """Recorte BGR da placa -> tensor (1, 48, 128, 1) em [0, 1]."""
    pronto = prepare_plate(crop, guess_vehicle_type(crop))
    return (to_gray(pronto).astype(np.float32) / 255.0)[None, :, :, None]


def decode_positions(probs: np.ndarray) -> PlateReading:
    """Probabilidades (7, 36) -> texto cru e confiança do caractere menos certo."""
    if probs.shape != (PLATE_LENGTH, len(ALPHABET)):
        raise ValueError(f"Saída do OCR com formato inesperado: {probs.shape}")
    indices = probs.argmax(axis=1)
    texto = "".join(ALPHABET[i] for i in indices)
    confianca = float(probs[np.arange(PLATE_LENGTH), indices].min())
    return PlateReading(text=texto, confidence=confianca)


def _load_predict(path: Path) -> Predict:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Modelo não encontrado: {path} (rode o treino em training/)")
    import keras  # importação tardia: só quem usa o modelo real paga o custo do TensorFlow

    modelo = keras.models.load_model(path, compile=False)
    return lambda x: np.asarray(modelo(x, training=False))


class TfPlateDetector:
    """PlateDetector real. Recebe a função de predição pronta (o modelo é carregado uma vez)."""

    def __init__(
        self,
        predict: Predict,
        threshold: float = 0.5,
        input_width: int = DETECTOR_INPUT_WIDTH,
    ) -> None:
        self._predict = predict
        self._threshold = threshold
        self._input_width = input_width

    @classmethod
    def from_file(cls, path: Path = DETECTOR_PATH, **kwargs) -> "TfPlateDetector":
        return cls(_load_predict(path), **kwargs)

    def detect(self, frame: np.ndarray) -> list[PlateDetection]:
        entrada, escala = detector_input(frame, self._input_width)
        prob = self._predict(entrada)[0, :, :, 0]
        return mask_to_detections(prob, escala, frame.shape, self._threshold)


class TfPlateReader:
    """PlateReader real. Devolve o texto cru (sem normalizar) e a confiança."""

    def __init__(self, predict: Predict) -> None:
        self._predict = predict

    @classmethod
    def from_file(cls, path: Path = OCR_PATH) -> "TfPlateReader":
        return cls(_load_predict(path))

    def read(self, crop: np.ndarray) -> PlateReading | None:
        if crop.ndim != 3 or crop.shape[0] < 2 or crop.shape[1] < 2:
            return None
        return decode_positions(self._predict(ocr_input(crop))[0])
