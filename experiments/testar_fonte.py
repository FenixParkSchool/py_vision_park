import logging
from contextlib import closing
from pathlib import Path

import cv2

from app.camera.source import CameraSource
from app.config import get_settings

logging.basicConfig(level=logging.INFO)

settings = get_settings()
saida = Path("experiments/frames")
saida.mkdir(parents=True, exist_ok=True)

camera = CameraSource(settings.camera_source, settings.sample_interval_seconds)
camera.open()
print("FPS informado:", camera.fps)

selecionados = 0
with closing(camera.frames()) as frames:
    for frame in frames:
        if selecionados == 0:
            print("Primeiro frame:", frame.shape, frame.dtype)
        cv2.imwrite(str(saida / f"frame_{selecionados:04d}.png"), frame)
        selecionados += 1

print("Frames selecionados:", selecionados)