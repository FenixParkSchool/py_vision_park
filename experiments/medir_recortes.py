import statistics
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from app.recognition.preprocessing import crop_region
from training.rds import find_images, load_sample

RAIZ = Path("data/rodosol")
SAIDA = Path("experiments/crops")
POR_GRUPO = 500   # imagens analisadas por pasta
SALVAR = 5        # recortes gravados por pasta, para olhar a olho nu

SAIDA.mkdir(parents=True, exist_ok=True)

grupos: dict[str, list[Path]] = defaultdict(list)
for caminho in find_images(RAIZ):
    grupos[caminho.relative_to(RAIZ).parts[0]].append(caminho)

for nome, caminhos in sorted(grupos.items()):
    passo = max(1, len(caminhos) // POR_GRUPO)
    amostra = caminhos[::passo][:POR_GRUPO]  # espalhadas pela pasta toda

    tamanhos_frame: Counter[tuple] = Counter()
    larguras: list[int] = []
    alturas: list[int] = []
    ilegiveis = fora = cortados = salvos = 0

    for caminho in amostra:
        gabarito = load_sample(caminho).annotation
        imagem = cv2.imread(str(caminho))
        if imagem is None:
            ilegiveis += 1
            continue
        tamanhos_frame[imagem.shape] += 1
        bbox = gabarito.bbox
        larguras.append(bbox.width)
        alturas.append(bbox.height)

        recorte = crop_region(imagem, bbox)
        if recorte is None:
            fora += 1
            continue
        if recorte.shape[:2] != (bbox.height, bbox.width):
            cortados += 1
        if salvos < SALVAR:
            cv2.imwrite(str(SAIDA / f"{nome}_{salvos}_{gabarito.plate}.png"), recorte)
            salvos += 1

    print(f"\n== {nome} ({len(amostra)} imagens) ==")
    print("Tamanhos de frame:", dict(tamanhos_frame))
    print("Ilegíveis:", ilegiveis, "| caixa fora do frame:", fora, "| caixa cortada na borda:", cortados)
    print(
        "Largura da placa (min/mediana/max):",
        min(larguras), statistics.median(larguras), max(larguras),
    )
    print(
        "Altura da placa  (min/mediana/max):",
        min(alturas), statistics.median(alturas), max(alturas),
    )