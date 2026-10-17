from collections import Counter
from pathlib import Path

from training.rds import find_images, load_sample

raiz = Path("data/rodosol")
imagens = find_images(raiz)
print("Imagens .jpg encontradas:", len(imagens))

layouts: Counter[str] = Counter()
tipos: Counter[str] = Counter()
erros: list[str] = []

for caminho in imagens:
    try:
        amostra = load_sample(caminho)
    except (FileNotFoundError, ValueError) as erro:
        erros.append(str(erro))
        continue
    layouts[amostra.annotation.layout] += 1
    tipos[amostra.annotation.vehicle_type] += 1

print("Por layout:", dict(layouts))
print("Por tipo:", dict(tipos))
print("Imagens com problema no gabarito:", len(erros))
for erro in erros[:10]:
    print(" -", erro)