import pytest

from app.schemas.recognition import BoundingBox
from training.evaluate import Record, char_hits, iou, summarize


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
