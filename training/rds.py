from dataclasses import dataclass

from app.schemas.recognition import BoundingBox

CAMPOS_OBRIGATORIOS = {"type", "plate", "layout", "corners"}


@dataclass(frozen=True)
class PlateAnnotation:
    """Gabarito de uma imagem do RodoSol-ALPR: o que existe e onde está a placa."""

    vehicle_type: str
    plate: str
    layout: str
    bbox: BoundingBox


def parse_annotation(texto: str) -> PlateAnnotation:
    campos: dict[str, str] = {}
    for linha in texto.splitlines():
        nome, separador, valor = linha.partition(":")
        if separador:
            campos[nome.strip()] = valor.strip()

    faltando = CAMPOS_OBRIGATORIOS - campos.keys()
    if faltando:
        raise ValueError(f"Anotação sem os campos: {sorted(faltando)}")

    try:
        pontos = [tuple(int(n) for n in par.split(",")) for par in campos["corners"].split()]
    except ValueError as erro:
        raise ValueError(f"Cantos inválidos: {campos['corners']!r}") from erro
    if len(pontos) != 4 or any(len(p) != 2 for p in pontos):
        raise ValueError(f"Esperados 4 cantos (x,y), recebidos: {campos['corners']!r}")

    xs = [x for x, _ in pontos]
    ys = [y for _, y in pontos]
    bbox = BoundingBox(x=min(xs), y=min(ys), width=max(xs) - min(xs), height=max(ys) - min(ys))
    return PlateAnnotation(campos["type"], campos["plate"], campos["layout"], bbox)