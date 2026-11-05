"""Demonstração: pipeline real (detector + OCR) numa imagem ou na webcam.

Imagem:  .venv\\Scripts\\python.exe -m experiments.demo_reconhecer data/rodosol/cars-me/img_004501.jpg
Webcam:  .venv\\Scripts\\python.exe -m experiments.demo_reconhecer 0      (Ctrl+C para sair)

Na imagem, grava experiments/crops/demo.png com a caixa e o texto desenhados.
Mostra a placa completa no terminal de propósito (é uma ferramenta de desenvolvimento).
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2

from app.camera.source import CameraSource
from app.recognition.plate_format import normalize_plate
from app.recognition.tf_models import TfPlateDetector, TfPlateReader
from app.services.pipeline import RecognitionPipeline

LIMIAR = 0.6

fonte = sys.argv[1] if len(sys.argv) > 1 else "data/rodosol/cars-me/img_004501.jpg"
pipeline = RecognitionPipeline(
    TfPlateDetector.from_file(), TfPlateReader.from_file(), camera_id="demo", min_confidence=LIMIAR
)


def mostrar(obs) -> None:
    placa = normalize_plate(obs.plate_text)
    print(
        f"placa crua={obs.plate_text} normalizada={placa.text} formato={placa.format} "
        f"det={obs.detection_confidence:.2f} ocr={obs.ocr_confidence:.2f} caixa={obs.bbox}"
    )


if fonte.isdigit():
    for obs in pipeline.process_stream(CameraSource(fonte, sample_interval_s=0.5).frames()):
        mostrar(obs)
else:
    img = cv2.imread(fonte)
    if img is None:
        sys.exit(f"Não consegui ler {fonte}")
    observacoes = pipeline.process_frame(img, datetime.now(timezone.utc))
    if not observacoes:
        print(f"Nenhuma placa acima de {LIMIAR}")
    for obs in observacoes:
        mostrar(obs)
        b = obs.bbox
        cv2.rectangle(img, (b.x, b.y), (b.x + b.width, b.y + b.height), (0, 255, 0), 2)
        cv2.putText(img, obs.plate_text, (b.x, max(20, b.y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    saida = Path("experiments/crops/demo.png")
    saida.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(saida), img)
    print("Imagem anotada em", saida)
