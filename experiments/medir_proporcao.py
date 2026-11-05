from collections import Counter
from pathlib import Path

from app.recognition.preprocessing import CAR_MIN_ASPECT
from training.rds import find_images, load_sample

RAIZ = Path("data/rodosol")

acertos: Counter[str] = Counter()
erros: Counter[str] = Counter()
piores: list[tuple[float, str, str]] = []

for caminho in find_images(RAIZ):
    a = load_sample(caminho).annotation
    proporcao = a.bbox.width / a.bbox.height
    palpite = "car" if proporcao >= CAR_MIN_ASPECT else "motorcycle"
    if palpite == a.vehicle_type:
        acertos[a.vehicle_type] += 1
    else:
        erros[a.vehicle_type] += 1
        piores.append((proporcao, a.vehicle_type, caminho.name))

print("Acertos:", dict(acertos))
print("Erros:", dict(erros))
for p, tipo, nome in sorted(piores)[:10]:
    print(f" - proporção {p:.2f}, gabarito {tipo}: {nome}")