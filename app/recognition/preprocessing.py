import numpy as np

import cv2

from app.schemas.recognition import BoundingBox

from typing import Literal


def crop_region(frame: np.ndarray, bbox: BoundingBox) -> np.ndarray | None:
    """Recorta a região da caixa, cortando o que passar das bordas do frame.

    Devolve None se nada da caixa estiver dentro do frame.
    O BoundingBox já garante x >= 0, y >= 0, largura > 0 e altura > 0.
    """
    altura, largura = frame.shape[:2]
    x1 = min(bbox.x, largura)
    y1 = min(bbox.y, altura)
    x2 = min(bbox.x + bbox.width, largura)
    y2 = min(bbox.y + bbox.height, altura)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]

VehicleType = Literal["car", "motorcycle"]

# Tamanho de entrada do OCR como (largura, altura), a ordem que o cv2.resize espera.
# Valor provisório de desenvolvimento, ainda não validado.
PLATE_SIZE = (128, 48)

# Acima desta proporção largura/altura, a caixa é tratada como carro. Provisório: será medido.
CAR_MIN_ASPECT = 2.0


def guess_vehicle_type(crop: np.ndarray) -> VehicleType:
    """Palpite por proporção: placa de carro é larga (1 linha), a de moto é quase quadrada (2 linhas)."""
    altura, largura = crop.shape[:2]
    return "car" if largura / altura >= CAR_MIN_ASPECT else "motorcycle"


def stack_two_lines(crop: np.ndarray) -> np.ndarray:
    """Corta a placa de duas linhas ao meio e cola as metades lado a lado (linha de cima à esquerda)."""
    altura = crop.shape[0]
    if altura < 2:
        raise ValueError("Recorte baixo demais para dividir em duas linhas")
    meio = altura // 2
    cima = crop[:meio]
    baixo = crop[altura - meio :]  # mesma altura que 'cima', mesmo se a altura for ímpar
    return np.hstack([cima, baixo])


def prepare_plate(crop: np.ndarray, vehicle_type: VehicleType) -> np.ndarray:
    """Deixa o recorte no formato de entrada do OCR: uma linha, tamanho fixo PLATE_SIZE."""
    if crop.ndim != 3 or crop.size == 0:
        raise ValueError("Recorte vazio ou sem 3 canais")
    if vehicle_type == "motorcycle":
        crop = stack_two_lines(crop)
    elif vehicle_type != "car":
        raise ValueError(f"Tipo de veículo desconhecido: {vehicle_type!r}")
    return cv2.resize(crop, PLATE_SIZE, interpolation=cv2.INTER_LINEAR)