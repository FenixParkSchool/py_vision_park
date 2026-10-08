import logging
import math
from urllib.parse import urlsplit

# log = logging.getLogger(__name__): cria o “diário” deste arquivo.
log = logging.getLogger(__name__)

# FPS assumido quando a fonte informa um valor inválido (webcams às vezes devolvem 0)
DEFAULT_FPS = 30.0


class CameraError(Exception):
    """Erro base de qualquer problema com a fonte de vídeo."""


class CameraOpenError(CameraError):
    """A fonte não pôde ser aberta."""


class CameraReadError(CameraError):
    """Uma fonte ao vivo parou de entregar frames."""


def parse_source(raw: str) -> int | str:
    """Só texto formado apenas por dígitos vira índice de webcam."""
    raw = raw.strip()
    if not raw:
        raise ValueError("A fonte da câmera não pode ser vazia")
    if raw.isdecimal():
        return int(raw)
    return raw


def describe_source(source: int | str) -> str:
    """Descrição segura para logs: nunca inclui usuário e senha de URLs."""
    if isinstance(source, int):
        return f"webcam #{source}"
    if "://" in source:
        partes = urlsplit(source)
        porta = f":{partes.port}" if partes.port else ""
        return f"{partes.scheme}://{partes.hostname}{porta}{partes.path}"
    return source


def compute_stride(fps: float, interval_s: float) -> int:
    """De quantos em quantos frames analisar (seleção por contagem de frames)."""
    if not math.isfinite(fps) or fps <= 0:
        log.warning("FPS inválido (%s); usando %s", fps, DEFAULT_FPS)
        fps = DEFAULT_FPS
    return max(1, round(fps * interval_s))