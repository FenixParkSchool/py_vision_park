import logging
import math
import threading
from collections.abc import Callable, Iterator
from typing import Any
from urllib.parse import urlsplit

import cv2
import numpy as np

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

class CameraSource:
    """Abre, lê e libera uma fonte de vídeo (arquivo, webcam ou URL)."""

    def __init__(
        self,
        source: str,
        sample_interval_s: float,
        capture_factory: Callable[[int | str], Any] = cv2.VideoCapture,
    ) -> None:
        self._source = parse_source(source)
        self._sample_interval_s = sample_interval_s
        self._capture_factory = capture_factory
        self._cap: Any = None
        self._stop = threading.Event()

    @property
    def is_live(self) -> bool:
        """Webcam e URL são 'ao vivo'; qualquer outra coisa é um arquivo."""
        return isinstance(self._source, int) or "://" in self._source

    @property
    def fps(self) -> float:
        if self._cap is None:
            raise CameraError("A fonte ainda não foi aberta")
        return float(self._cap.get(cv2.CAP_PROP_FPS))

    def open(self) -> None:
        if self._cap is not None:
            return
        descricao = describe_source(self._source)
        cap = self._capture_factory(self._source)
        if not cap.isOpened():
            cap.release()
            raise CameraOpenError(f"Não foi possível abrir a fonte: {descricao}")
        self._cap = cap
        log.info("Fonte aberta: %s", descricao)

    def frames(self) -> Iterator[np.ndarray]:
        self.open()
        cap = self._cap
        try:
            passo = compute_stride(self.fps, self._sample_interval_s)
            indice = 0
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None or frame.size == 0:
                    if self.is_live:
                        raise CameraReadError(
                            f"A fonte parou de entregar frames: {describe_source(self._source)}"
                        )
                    return  # arquivo: fim do vídeo
                if indice % passo == 0:
                    yield frame
                indice += 1
        finally:
            self.close()

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            log.info("Fonte liberada")