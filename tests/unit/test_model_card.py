from app.recognition.model_card import card_path, describe, read_card, write_card


def test_ficha_grava_le_e_descreve(tmp_path):
    modelo = tmp_path / "ocr.keras"
    modelo.write_bytes(b"pesos")
    destino = write_card(modelo, "ocr", {"epochs": 1}, {"val_char_accuracy": 0.9}, {"train": 10})
    assert destino == card_path(modelo) == tmp_path / "ocr.json"
    ficha = read_card(modelo)
    assert ficha["kind"] == "ocr" and ficha["metrics"]["val_char_accuracy"] == 0.9
    assert len(ficha["sha256"]) == 64
    assert ficha["sha256"][:12] in describe(modelo)


def test_sem_ficha(tmp_path):
    modelo = tmp_path / "x.keras"
    assert read_card(modelo) is None
    assert "sem ficha" in describe(modelo)
