"""Avalia OCRs de placa PRONTOS com o nosso detector e o nosso `training.evaluate`.

Uso (ambiente separado, ver .venv-pretrained):
    .venv-pretrained\\Scripts\\python.exe -m experiments.avaliar_ocr_pronto <candidato> [--limit 500]

Candidatos:
    fpo-mobilevit   fast-plate-ocr, global-plates-mobile-vit-v2-model (MIT)
    fpo-cct-s-v2    fast-plate-ocr, cct-s-v2-global-model (MIT)
    fpo-cct-xs-v2   fast-plate-ocr, cct-xs-v2-global-model (MIT)
    accessalpr      AccessALPR (GPL-2.0), so o leitor (7 posicoes). Precisa de torch/torchvision
                    e de dois caminhos: --accessalpr-repo (clone do github.com/glefundes/AccessALPR)
                    e --accessalpr-weights (reader-w.pth do zip do Google Drive do README).

Adaptadores: recebem recorte BGR (padrao do OpenCV), convertem para o modo de cor do
modelo (cinza ou RGB, lido do plate_config), removem hifen/espaco/padding e devolvem
o texto cru em maiusculas. Nao corrige caracteres ambiguos. Grava models/avaliacao_<cand>.json.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from app.recognition.base import PlateReading

FPO_MODELOS = {
    "fpo-mobilevit": "global-plates-mobile-vit-v2-model",
    "fpo-cct-s-v2": "cct-s-v2-global-model",
    "fpo-cct-xs-v2": "cct-xs-v2-global-model",
}


def limpar(texto: str) -> str:
    """Maiusculas, sem hifen/espaco/padding ('_'). Texto cru, sem corrigir caracteres."""
    return "".join(c for c in texto.upper() if c.isalnum())


class FastPlateOcrReader:
    """PlateReader sobre fast-plate-ocr (ONNX Runtime, CPU). Modelo carregado uma vez."""

    def __init__(self, nome_modelo: str) -> None:
        from fast_plate_ocr import LicensePlateRecognizer

        self._rec = LicensePlateRecognizer(nome_modelo, device="cpu")
        self._cinza = "gray" in str(self._rec.config.image_color_mode).lower()

    def read(self, crop: np.ndarray) -> PlateReading | None:
        if crop.ndim != 3 or crop.shape[0] < 2 or crop.shape[1] < 2:
            return None
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY if self._cinza else cv2.COLOR_BGR2RGB)
        pred = self._rec.run_one(img, return_confidence=True)
        texto = limpar(pred.plate)
        if not texto:
            return None
        probs = pred.char_probs
        conf = float(np.mean(probs[probs > 0])) if probs is not None and np.any(probs > 0) else 0.0
        return PlateReading(text=texto, confidence=conf)


class AccessAlprReader:
    """PlateReader sobre o leitor do AccessALPR (EfficientNet-B0, 7 posicoes, 36 classes)."""

    def __init__(self, repo: Path, pesos: Path) -> None:
        import sys

        import torch

        sys.path.insert(0, str(repo))
        from reader import efficientnet, PlateReader as modulo  # noqa: F401 (codigo do repo, GPL)

        # evita baixar pesos ImageNet: os pesos do leitor sobrescrevem tudo logo depois
        efficientnet.EfficientNet.from_pretrained = classmethod(
            lambda cls, nome, num_classes=1000: cls.from_name(
                nome, override_params={"num_classes": num_classes}
            )
        )
        self._torch = torch
        self._net = modulo.PlateReader()
        self._net.load_state_dict(torch.load(pesos, map_location="cpu"))
        self._net.eval()

    def read(self, crop: np.ndarray) -> PlateReading | None:
        from PIL import Image

        if crop.ndim != 3 or crop.shape[0] < 2 or crop.shape[1] < 2:
            return None
        x = self._net.transf(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        with self._torch.no_grad():
            probs = self._torch.sigmoid(self._net(x[None]))[0]  # (7, 36)
        conf, idx = probs.max(dim=1)
        from reader import utils

        texto = limpar(utils.output2word(idx.tolist()))
        return PlateReading(text=texto, confidence=float(conf.mean())) if texto else None


def criar_leitor(candidato: str, args=None):
    if candidato in FPO_MODELOS:
        return FastPlateOcrReader(FPO_MODELOS[candidato])
    if candidato == "accessalpr":
        return AccessAlprReader(args.accessalpr_repo, args.accessalpr_weights)
    raise SystemExit(f"candidato desconhecido: {candidato}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidato", choices=sorted([*FPO_MODELOS, "accessalpr"]))
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--detector", type=Path, default=Path("models/detector.keras"))
    ap.add_argument("--accessalpr-repo", type=Path)
    ap.add_argument("--accessalpr-weights", type=Path)
    args = ap.parse_args()

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
        TfPlateDetector.from_file(args.detector), criar_leitor(args.candidato, args), teste
    )
    resultado = {
        "candidato": args.candidato,
        "imagens": len(teste),
        "tempos": tempos,
        "grupos": summarize(records),
        "curva_confianca": confidence_curve(records),
    }
    saida = Path(f"models/avaliacao_{args.candidato}.json")
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    for nome, g in resultado["grupos"].items():
        print(
            f"{nome:18s} n={g['n']:4d} det={g['deteccao']:.3f} ocr_inteira={g['ocr_placa_inteira']:.3f} "
            f"ocr_car={g['ocr_por_caractere']:.3f} e2e={g['e2e_placa_inteira']:.3f}"
        )
    print("tempos ms/recorte:", tempos)


if __name__ == "__main__":
    main()
