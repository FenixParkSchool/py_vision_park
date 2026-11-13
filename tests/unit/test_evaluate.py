import pytest

from app.schemas.recognition import BoundingBox
from training.evaluate import Record, char_hits, iou, position_hits, summarize


def test_iou():
    a = BoundingBox(x=0, y=0, width=10, height=10)
    assert iou(a, a) == 1.0
    assert iou(a, BoundingBox(x=20, y=20, width=5, height=5)) == 0.0
    assert iou(a, BoundingBox(x=5, y=0, width=10, height=10)) == pytest.approx(50 / 150)


def test_char_hits():
    assert char_hits("ABC1D23", "ABC1D23") == 7
    assert char_hits("ABCID23", "ABC1D23") == 6
    assert char_hits(None, "ABC1D23") == 0


def test_summarize_por_grupo_e_total():
    recs = [
        Record("car/Mercosul", "ABC1D23", True, 0.9, "ABC1D23", "ABC1D23"),
        Record("car/Mercosul", "ABC1D23", True, 0.8, "ABCID23", "ABCID23"),  # I no lugar do 1
        Record("motorcycle/Brazilian", "ABC1234", False, 0.1, "ABC1234", None),
    ]
    r = summarize(recs)
    m = r["car/Mercosul"]
    assert m["n"] == 2 and m["deteccao"] == 1.0
    assert m["ocr_placa_inteira"] == 0.5
    assert m["ocr_por_caractere"] == round(13 / 14, 4)
    assert m["e2e_formato_valido"] == 0.5  # ABCID23 é inválida
    assert r["motorcycle/Brazilian"]["deteccao"] == 0.0
    assert r["motorcycle/Brazilian"]["e2e_placa_inteira"] == 0.0
    assert r["todos"]["n"] == 3


def test_position_hits_comprimento_diferente_conta_erro():
    assert position_hits("ABC1D23", "ABC1D23") == [1] * 7
    assert position_hits("ABCID23", "ABC1D23") == [1, 1, 1, 0, 1, 1, 1]
    assert position_hits("ABC", "ABC1D23") == [1, 1, 1, 0, 0, 0, 0]
    assert position_hits(None, "ABC1D23") == [0] * 7


def test_summarize_por_posicao():
    recs = [
        Record("car/Mercosul", "ABC1D23", True, 0.9, "ABC1D23", "ABC1D23"),
        Record("car/Mercosul", "ABC1D23", True, 0.8, "ABCID23", None),
    ]
    m = summarize(recs)["car/Mercosul"]
    assert m["ocr_por_posicao"] == [1, 1, 1, 0.5, 1, 1, 1]
    assert m["e2e_por_posicao"] == [0.5] * 7


def test_confidence_curve():
    from training.evaluate import confidence_curve

    recs = [
        Record("g", "ABC1D23", True, 0.9, "ABC1D23", "ABC1D23", 0.95),
        Record("g", "ABC1D23", True, 0.9, "ABC1D23", "ABC1D24", 0.55),
        Record("g", "ABC1D23", True, 0.9, "ABC1D23", "ABC1D23", 0.30),
        Record("g", "ABC1D23", False, 0.0, "ABC1D23", None, 0.0),
    ]
    c = {x["limiar"]: x for x in confidence_curve(recs, [0.0, 0.5, 0.9, 0.99])}
    assert c[0.0]["cobertura"] == 0.75 and c[0.0]["precisao"] == round(2 / 3, 4)
    assert c[0.5]["cobertura"] == 0.5 and c[0.5]["precisao"] == 0.5
    assert c[0.9]["cobertura"] == 0.25 and c[0.9]["precisao"] == 1.0
    assert c[0.99]["enviadas"] == 0 and c[0.99]["precisao"] is None


def test_limiares_padrao_incluem_finos():
    from training.evaluate import THRESHOLDS

    assert {0.0, 0.5, 0.9, 0.95, 0.97, 0.98, 0.99} <= set(THRESHOLDS)
