from pathlib import Path

import cv2

from app.recognition.preprocessing import crop_region, prepare_plate
from training.rds import find_images, load_sample

RAIZ = Path("data/rodosol")
SAIDA = Path("experiments/preparados")
SAIDA.mkdir(parents=True, exist_ok=True)

vistos: dict[str, int] = {}
for caminho in find_images(RAIZ)[::500]:
    a = load_sample(caminho).annotation
    nome = caminho.relative_to(RAIZ).parts[0]
    if vistos.get(nome, 0) >= 5:
        continue
    imagem = cv2.imread(str(caminho))
    recorte = crop_region(imagem, a.bbox)
    pronto = prepare_plate(recorte, a.vehicle_type)
    cv2.imwrite(str(SAIDA / f"{nome}_{vistos.get(nome, 0)}_{a.plate}.png"), pronto)
    vistos[nome] = vistos.get(nome, 0) + 1
print("Gravados em", SAIDA)