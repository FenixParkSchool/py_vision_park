import numpy as np

from app.schemas.recognition import BoundingBox


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