"""Avalia detector + OCR no conjunto de teste do RodoSol-ALPR, por tipo de veículo e layout.

Uso:  .venv\\Scripts\\python.exe -m training.evaluate [--limit 500]
Saída: JSON no terminal e em models/avaliacao.json (guardar junto das fichas dos modelos).

Métricas por grupo (tipo de veículo/layout):
- detecção: a melhor caixa tem IoU >= 0.5 com o gabarito;
- OCR isolado: lê o recorte do gabarito (mede só o OCR);
- ponta a ponta: lê o recorte da caixa detectada;
- formato válido: o texto lido passa em `normalize_plate` (Mercosul/antiga).
As funções de contagem são puras e testadas sem modelo nem dataset.
"""

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2

from app.recognition.plate_format import normalize_plate
from app.recognition.preprocessing import crop_region
from app.schemas.recognition import BoundingBox
from training.rds import find_images, load_sample
from training.split import split_paths
from training.train_detector import RAIZ

IOU_MIN = 0.5


def iou(a: BoundingBox, b: BoundingBox) -> float:
    x1, y1 = max(a.x, b.x), max(a.y, b.y)
    x2, y2 = min(a.x + a.width, b.x + b.width), min(a.y + a.height, b.y + b.height)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    uniao = a.width * a.height + b.width * b.height - inter
    return inter / uniao if uniao > 0 else 0.0


def char_hits(pred: str | None, true: str) -> int:
    """Quantos caracteres batem na mesma posição (texto ausente = 0)."""
    if not pred:
        return 0
    return sum(p == t for p, t in zip(pred, true))


def position_hits(pred: str | None, true: str) -> list[int]:
    """1/0 por posição do gabarito; texto ausente ou posição sem par = erro."""
    pred = pred or ""
    return [int(i < len(pred) and pred[i] == c) for i, c in enumerate(true)]


@dataclass(frozen=True)
class Record:
    group: str  # ex.: "car/Mercosul"
    plate: str  # gabarito
    detected: bool  # alguma caixa com IoU >= 0.5
    iou: float
    gt_text: str | None  # leitura do recorte do gabarito
    e2e_text: str | None  # leitura do recorte detectado (None se não detectou/ilegível)
    e2e_confidence: float = 0.0  # confiança da leitura ponta a ponta (0 se não houve leitura)


def summarize(records: list[Record]) -> dict[str, dict]:
    """Agrupa por `group` e por "todos"; devolve contagens e taxas em [0, 1]."""
    grupos: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        grupos[r.group].append(r)
        grupos["todos"].append(r)

    def taxa(n: int, d: int) -> float:
        return round(n / d, 4) if d else 0.0

    saida: dict[str, dict] = {}
    for nome, lista in sorted(grupos.items()):
        n = len(lista)

        def por_posicao(campo: str) -> list[float]:
            tot = [0] * 7
            for r in lista:
                for i, h in enumerate(position_hits(getattr(r, campo), r.plate)[:7]):
                    tot[i] += h
            return [taxa(t, n) for t in tot]

        saida[nome] = {
            "n": n,
            "deteccao": taxa(sum(r.detected for r in lista), n),
            "ocr_placa_inteira": taxa(sum(r.gt_text == r.plate for r in lista), n),
            "ocr_por_caractere": taxa(sum(char_hits(r.gt_text, r.plate) for r in lista), n * 7),
            "ocr_por_posicao": por_posicao("gt_text"),
            "e2e_placa_inteira": taxa(sum(r.e2e_text == r.plate for r in lista), n),
            "e2e_por_posicao": por_posicao("e2e_text"),
            "e2e_formato_valido": taxa(
                sum(bool(r.e2e_text) and normalize_plate(r.e2e_text).is_valid for r in lista), n
            ),
        }
    return saida


THRESHOLDS = [i / 10 for i in range(10)] + [0.95, 0.97, 0.98, 0.99]


def confidence_curve(records: list[Record], thresholds=THRESHOLDS) -> list[dict]:
    """Curva confiança x acerto para escolher o filtro.

    Por limiar: `cobertura` = fração das imagens cuja leitura ponta a ponta seria enviada
    (há texto e confiança >= limiar); `precisao` = placa inteira correta entre as enviadas
    (None se nenhuma foi enviada).
    """
    n = len(records)
    curva = []
    for t in thresholds:
        enviadas = [r for r in records if r.e2e_text and r.e2e_confidence >= t]
        certas = sum(r.e2e_text == r.plate for r in enviadas)
        curva.append(
            {
                "limiar": round(t, 2),
                "enviadas": len(enviadas),
                "cobertura": round(len(enviadas) / n, 4) if n else 0.0,
                "precisao": round(certas / len(enviadas), 4) if enviadas else None,
            }
        )
    return curva


def evaluate(detector, reader, paths: list[Path]) -> tuple[list[Record], dict[str, float]]:
    records: list[Record] = []
    tempos = {"detector": 0.0, "ocr": 0.0}
    chamadas = {"detector": 0, "ocr": 0}
    for caminho in paths:
        a = load_sample(caminho).annotation
        frame = cv2.imread(str(caminho))
        t0 = time.perf_counter()
        deteccoes = detector.detect(frame)
        tempos["detector"] += time.perf_counter() - t0
        chamadas["detector"] += 1

        melhor = max(deteccoes, key=lambda d: iou(d.bbox, a.bbox), default=None)
        melhor_iou = iou(melhor.bbox, a.bbox) if melhor else 0.0
        detectou = melhor_iou >= IOU_MIN

        def ler(caixa: BoundingBox):
            """(texto, confiança) da leitura, ou (None, 0.0)."""
            recorte = crop_region(frame, caixa)
            if recorte is None:
                return None, 0.0
            t = time.perf_counter()
            leitura = reader.read(recorte)
            tempos["ocr"] += time.perf_counter() - t
            chamadas["ocr"] += 1
            return (leitura.text, leitura.confidence) if leitura else (None, 0.0)

        gt_text, _ = ler(a.bbox)
        e2e_text, e2e_conf = ler(melhor.bbox) if detectou else (None, 0.0)
        records.append(
            Record(
                group=f"{a.vehicle_type}/{a.layout}",
                plate=a.plate,
                detected=detectou,
                iou=melhor_iou,
                gt_text=gt_text,
                e2e_text=e2e_text,
                e2e_confidence=e2e_conf,
            )
        )
    medios = {f"{k}_ms_medio": round(1000 * tempos[k] / max(1, chamadas[k]), 2) for k in tempos}
    return records, medios


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="máximo de imagens (0 = todas do teste)")
    ap.add_argument("--detector", type=Path, default=Path("models/detector.keras"))
    ap.add_argument("--ocr", type=Path, default=Path("models/ocr.keras"))
    ap.add_argument("--out", type=Path, default=Path("models/avaliacao.json"))
    args = ap.parse_args()

    from app.recognition.tf_models import TfPlateDetector, TfPlateReader

    teste = split_paths(find_images(RAIZ))["test"]
    if args.limit:
        passo = max(1, len(teste) // args.limit)
        teste = teste[::passo][: args.limit]
    records, tempos = evaluate(
        TfPlateDetector.from_file(args.detector), TfPlateReader.from_file(args.ocr), teste
    )
    resultado = {
        "imagens": len(teste),
        "tempos": tempos,
        "grupos": summarize(records),
        "curva_confianca": confidence_curve(records),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
