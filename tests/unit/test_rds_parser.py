import pytest

from training.rds import parse_annotation

EXEMPLO = """type: car
plate: MSE9E82
layout: Mercosur
corners: 816,370 944,371 945,411 817,411
"""


def test_le_os_campos_de_texto():
    anotacao = parse_annotation(EXEMPLO)
    assert anotacao.vehicle_type == "car"
    assert anotacao.plate == "MSE9E82"
    assert anotacao.layout == "Mercosur"


def test_caixa_envolve_os_quatro_cantos():
    bbox = parse_annotation(EXEMPLO).bbox
    assert (bbox.x, bbox.y, bbox.width, bbox.height) == (816, 370, 129, 41)


@pytest.mark.parametrize(
    "cantos, esperado",
    [
        ("804,321 926,322 926,360 807,359", (804, 321, 122, 39)),
        ("855,318 978,319 977,358 856,357", (855, 318, 123, 40)),
        ("926,252 1030,252 1030,285 926,286", (926, 252, 104, 34)),
    ],
)
def test_outros_exemplos_reais(cantos, esperado):
    bbox = parse_annotation(f"type: car\nplate: ABC1D23\nlayout: Mercosur\ncorners: {cantos}\n").bbox
    assert (bbox.x, bbox.y, bbox.width, bbox.height) == esperado


def test_campo_ausente_levanta_erro():
    with pytest.raises(ValueError, match="plate"):
        parse_annotation("type: car\nlayout: Mercosur\ncorners: 1,1 5,1 5,5 1,5\n")


@pytest.mark.parametrize("cantos", ["1,1 5,1 5,5", "a,b c,d e,f g,h", "1,1,1 5,1 5,5 1,5"])
def test_cantos_invalidos_levantam_erro(cantos):
    with pytest.raises(ValueError):
        parse_annotation(f"type: car\nplate: ABC1D23\nlayout: Mercosur\ncorners: {cantos}\n")