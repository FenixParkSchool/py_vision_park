"""Avalia o leitor INTEGRADO (app.recognition.fpo_reader.FastPlateReader) com o nosso detector.

Uso: .venv\Scripts\python -m experiments.avaliar_fpo [--limit 0] [--model cct-s-v2-global-model]
Grava models/avaliacao_fpo_integrado.json.
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="cct-s-v2-global-model")
    ap.add_argument("--detector", type=Path, default=Path("models/detector.keras"))
    args = ap.parse_args()

    from app.recognition.fpo_reader import FastPlateReader
    from app.recognition.tf_models import TfPlateDetector
    from training.evaluate import confidence_curve, evaluate, summarize
    from training.rds import find_images
    from training.split import split_paths
    from training.train_detector import RAIZ

    teste = split_paths(find_images(RAIZ))["test"]
    if args.limit:
        passo = max(1, len(teste) // args.limit)
        teste = teste[::passo][: args.limit]
    records, tempos = evaluate(
        TfPlateDetector.from_file(args.detector), FastPlateReader.from_model(args.model), teste
    )
    resultado = {
        "modelo_ocr": args.model,
        "imagens": len(teste),
        "tempos": tempos,
        "grupos": summarize(records),
        "curva_confianca": confidence_curve(records),
    }
    saida = Path("models/avaliacao_fpo_integrado.json")
    saida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    for nome, g in resultado["grupos"].items():
        print(
            f"{nome:18s} n={g['n']:4d} det={g['deteccao']:.3f} ocr_inteira={g['ocr_placa_inteira']:.3f} "
            f"e2e={g['e2e_placa_inteira']:.3f}"
        )
    print("tempos ms/recorte:", tempos)
    for c in resultado["curva_confianca"]:
        print(c)


if __name__ == "__main__":
    main()
