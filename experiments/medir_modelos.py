"""Mede detector e OCR num subconjunto do split de teste do RodoSol (imagens nunca vistas no treino).

Uso: .venv\\Scripts\\python.exe -m experiments.medir_modelos [por_pasta=100] [limiar=0.6]
"""

import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2

from app.recognition.plate_format import normalize_plate
from app.recognition.preprocessing import crop_region, guess_vehicle_type
from app.recognition.tf_models import TfPlateDetector, TfPlateReader
from training.rds import find_images, load_sample
from training.split import split_paths
from training.train_detector import RAIZ, pick

POR_PASTA = int(sys.argv[1]) if len(sys.argv) > 1 else 100
LIMIAR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6


def iou(a, b) -> float:
    x1, y1 = max(a.x, b.x), max(a.y, b.y)
    x2 = min(a.x + a.width, b.x + b.width)
    y2 = min(a.y + a.height, b.y + b.height)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / (a.width * a.height + b.width * b.height - inter)


detector = TfPlateDetector.from_file()
leitor = TfPlateReader.from_file()
c: dict[str, defaultdict] = defaultdict(lambda: defaultdict(float))
t_det = t_ocr = 0.0

for caminho in pick(split_paths(find_images(RAIZ))["test"], POR_PASTA):
    a = load_sample(caminho).annotation
    m = c[caminho.parent.name]
    img = cv2.imread(str(caminho))
    m["n"] += 1

    # OCR isolado, na caixa do gabarito
    t = time.perf_counter()
    lido = leitor.read(crop_region(img, a.bbox))
    t_ocr += time.perf_counter() - t
    m["ocr_placa"] += lido.text == a.plate
    m["ocr_char"] += sum(x == y for x, y in zip(lido.text, a.plate)) / 7

    # Detector e cadeia completa (detecção mais confiante -> OCR)
    t = time.perf_counter()
    deteccoes = sorted(detector.detect(img), key=lambda d: d.confidence, reverse=True)
    t_det += time.perf_counter() - t
    m["falsos"] += sum(iou(d.bbox, a.bbox) < 0.5 for d in deteccoes)
    if not deteccoes or iou(deteccoes[0].bbox, a.bbox) < 0.5:
        continue
    m["det_ok"] += 1
    m["iou"] += iou(deteccoes[0].bbox, a.bbox)
    recorte = crop_region(img, deteccoes[0].bbox)
    m["tipo_ok"] += guess_vehicle_type(recorte) == a.vehicle_type
    final = leitor.read(recorte)
    m["e2e"] += final.text == a.plate
    m["valida"] += normalize_plate(final.text).is_valid
    if final.confidence >= LIMIAR:
        m["aceitas"] += 1
        m["aceitas_ok"] += final.text == a.plate

total: defaultdict = defaultdict(float)
for pasta, m in sorted(c.items()):
    for k, v in m.items():
        total[k] += v
for nome, m in [*sorted(c.items()), ("TOTAL", total)]:
    n, ok = m["n"], max(m["det_ok"], 1)
    print(
        f"{nome:15s} n={n:.0f} | det recall={m['det_ok'] / n:.1%} IoU={m['iou'] / ok:.2f} "
        f"falsos={m['falsos']:.0f} tipo={m['tipo_ok'] / ok:.1%} | OCR(gabarito) placa={m['ocr_placa'] / n:.1%} "
        f"char={m['ocr_char'] / n:.1%} | ponta-a-ponta={m['e2e'] / n:.1%} formato_valido={m['valida'] / ok:.1%} "
        f"| conf>={LIMIAR}: aceitas={m['aceitas'] / n:.1%} acerto_das_aceitas={m['aceitas_ok'] / max(m['aceitas'], 1):.1%}"
    )
print(f"tempo médio: detector {t_det / total['n'] * 1000:.0f} ms/frame, OCR {t_ocr / total['n'] * 1000:.0f} ms/recorte (CPU)")
