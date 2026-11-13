"""Exporta recortes de placa do RodoSol-ALPR no formato de dados do fast-plate-ocr (ajuste fino).

Uso:  python -m training.exportar_fpo [--saida data/fpo] [--margem 0.08]
Saída: data/fpo/{train,val,test}/*.png e data/fpo/{train,val,test}.csv com as colunas
       image_path (relativo ao CSV), plate_text  -- formato lido por `fast-plate-ocr train`.

Os recortes usam a caixa do gabarito com uma pequena margem (o leitor recebe, em produção,
o recorte do nosso detector, que nunca é perfeito). Moto (placa de duas linhas) é exportada
como está: o fast-plate-ocr lê a placa inteira, sem cortar as linhas.
A divisão treino/val/teste é a mesma do resto do projeto (training/split.py).
"""

import argparse
import csv
from pathlib import Path

import cv2

from app.recognition.preprocessing import crop_region
from app.schemas.recognition import BoundingBox
from training.rds import find_images, load_sample
from training.split import split_paths
from training.train_detector import RAIZ


def expand_box(box: BoundingBox, margem: float, largura: int, altura: int) -> BoundingBox:
    """Aumenta a caixa em `margem` (fração do tamanho) de cada lado, sem passar da imagem."""
    dx, dy = int(round(box.width * margem)), int(round(box.height * margem))
    x1, y1 = max(0, box.x - dx), max(0, box.y - dy)
    x2, y2 = min(largura, box.x + box.width + dx), min(altura, box.y + box.height + dy)
    return BoundingBox(x=x1, y=y1, width=max(1, x2 - x1), height=max(1, y2 - y1))


def nome_saida(caminho: Path) -> str:
    """Nome único e estável: pasta do RodoSol + nome da imagem (ex.: cars-br_img_000123.png)."""
    return f"{caminho.parent.name}_{caminho.stem}.png"


def exportar(paths: list[Path], pasta: Path, margem: float) -> int:
    pasta.mkdir(parents=True, exist_ok=True)
    linhas = 0
    with open(pasta.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "plate_text"])
        for caminho in paths:
            a = load_sample(caminho).annotation
            img = cv2.imread(str(caminho))
            if img is None:
                continue
            altura, largura = img.shape[:2]
            recorte = crop_region(img, expand_box(a.bbox, margem, largura, altura))
            if recorte is None:
                continue
            nome = nome_saida(caminho)
            cv2.imwrite(str(pasta / nome), recorte)
            w.writerow([f"{pasta.name}/{nome}", a.plate])
            linhas += 1
    return linhas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", type=Path, default=Path("data/fpo"))
    ap.add_argument("--margem", type=float, default=0.08)
    args = ap.parse_args()
    divisao = split_paths(find_images(RAIZ))
    for nome in ("train", "val", "test"):
        n = exportar(divisao[nome], args.saida / nome, args.margem)
        print(f"{nome}: {n} recortes -> {args.saida / (nome + '.csv')}")


if __name__ == "__main__":
    main()
