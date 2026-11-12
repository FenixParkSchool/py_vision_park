"""Diagnóstico do OCR (só leitura): onde e por que ele erra. Não altera modelos nem dataset.

Uso:  .venv\\Scripts\\python.exe -m experiments.diagnostico_ocr [--n-test 800] [--n-train 800]
Lê o recorte do GABARITO (sem detector), para medir só o OCR.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from app.recognition.preprocessing import crop_region
from app.recognition.tf_models import ALPHABET, ocr_input
from training.rds import find_images, load_sample
from training.split import split_paths
from training.train_detector import RAIZ


def carregar(paths, n):
    passo = max(1, len(paths) // n)
    xs, placas, grupos = [], [], []
    for p in paths[::passo][:n]:
        a = load_sample(p).annotation
        rec = crop_region(cv2.imread(str(p)), a.bbox)
        if rec is None or len(a.plate) != 7:
            continue
        xs.append(ocr_input(rec)[0])
        placas.append(a.plate)
        grupos.append(f"{a.vehicle_type}/{a.layout}")
    return np.stack(xs), placas, grupos


def classe(c):
    return "L" if c.isalpha() else "D"


def analisar(modelo, xs, placas):
    probs = modelo.predict(xs, batch_size=64, verbose=0)  # (n, 7, 36)
    idx = probs.argmax(axis=2)
    pred = ["".join(ALPHABET[i] for i in linha) for linha in idx]
    conf = probs.max(axis=2)
    acertos = np.array([[p[i] == t[i] for i in range(7)] for p, t in zip(pred, placas)])
    erros_por_placa = (~acertos).sum(axis=1)
    conf_min = conf.min(axis=1)
    inteira = erros_por_placa == 0
    confusoes = Counter(
        (t[i], p[i]) for p, t in zip(pred, placas) for i in range(7) if p[i] != t[i]
    )
    # erro de "tipo": esperava dígito e leu letra (ou o contrário)
    tipo = sum(classe(t[i]) != classe(p[i]) for p, t in zip(pred, placas) for i in range(7) if p[i] != t[i])
    total_erros = int((~acertos).sum())
    return {
        "n": len(placas),
        "char_acc": round(float(acertos.mean()), 4),
        "placa_inteira": round(float(inteira.mean()), 4),
        "acc_por_posicao": [round(float(x), 3) for x in acertos.mean(axis=0)],
        "placas_por_n_erros": {str(k): int((erros_por_placa == k).sum()) for k in range(8)},
        "erros_de_tipo_letra_digito": f"{tipo}/{total_erros}",
        "top_confusoes_verdade_leu": [[f"{t}->{p}", n] for (t, p), n in confusoes.most_common(12)],
        "conf_min_media_placa_certa": round(float(conf_min[inteira].mean()), 3) if inteira.any() else None,
        "conf_min_media_placa_errada": round(float(conf_min[~inteira].mean()), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=800)
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--ocr", type=Path, default=Path("models/ocr.keras"))
    args = ap.parse_args()

    import keras

    modelo = keras.saving.load_model(args.ocr)
    divisao = split_paths(find_images(RAIZ))
    saida = {}
    for nome, n in (("treino", args.n_train), ("teste", args.n_test)):
        xs, placas, grupos = carregar(divisao["train" if nome == "treino" else "test"], n)
        saida[nome] = analisar(modelo, xs, placas)
        if nome == "teste":
            por_grupo = {}
            for g in sorted(set(grupos)):
                ids = [i for i, x in enumerate(grupos) if x == g]
                por_grupo[g] = analisar(modelo, xs[ids], [placas[i] for i in ids])["char_acc"]
            saida["teste_char_acc_por_grupo"] = por_grupo
    print(json.dumps(saida, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
