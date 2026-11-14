from pathlib import Path
from types import SimpleNamespace

import fast_plate_ocr
import numpy as np
import pytest

from app.recognition.fpo_reader import FastPlateReader, caminho_plate_config


def test_caminho_plate_config_segue_a_convencao_do_fast_plate_ocr():
    assert caminho_plate_config(Path("models/ocr_br.onnx")) == Path("models/ocr_br_plate_config.yaml")


def test_from_model_com_onnx_local_usa_o_arquivo_e_a_config_ao_lado(tmp_path, monkeypatch):
    onnx = tmp_path / "ocr_br.onnx"
    onnx.write_bytes(b"x")
    caminho_plate_config(onnx).write_text("x")
    recebido = {}

    def falso(*args, **kwargs):
        recebido.update(args=args, kwargs=kwargs)
        return SimpleNamespace(config=SimpleNamespace(image_color_mode="rgb"))

    monkeypatch.setattr(fast_plate_ocr, "LicensePlateRecognizer", falso)
    leitor = FastPlateReader.from_model(str(onnx))

    assert recebido["args"] == ()
    assert recebido["kwargs"]["onnx_model_path"] == onnx
    assert recebido["kwargs"]["plate_config_path"] == caminho_plate_config(onnx)
    assert leitor.model_name == str(onnx)


def test_from_model_com_onnx_sem_config_levanta_erro_claro(tmp_path):
    onnx = tmp_path / "ocr_br.onnx"
    onnx.write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="plate_config"):
        FastPlateReader.from_model(str(onnx))


def test_from_model_com_nome_do_hub_passa_o_nome(monkeypatch):
    recebido = {}

    def falso(*args, **kwargs):
        recebido.update(args=args, kwargs=kwargs)
        return SimpleNamespace(config=SimpleNamespace(image_color_mode="rgb"))

    monkeypatch.setattr(fast_plate_ocr, "LicensePlateRecognizer", falso)
    FastPlateReader.from_model("cct-s-v2-global-model")
    assert recebido["args"] == ("cct-s-v2-global-model",)


class FakeRec:
    def __init__(self, plate, probs, mode="grayscale"):
        self.config = SimpleNamespace(image_color_mode=mode)
        self._pred = SimpleNamespace(plate=plate, char_probs=probs)
        self.last = None

    def run_one(self, img, return_confidence=False):
        self.last = img
        return self._pred


def crop(h=20, w=60):
    return np.zeros((h, w, 3), np.uint8)


def test_limpa_texto_e_media_confianca_sem_padding():
    rec = FakeRec("abc-1d 23__", np.array([0.9, 0.8, 1.0, 0.7, 0.9, 0.9, 0.9, 0, 0, 0, 0]))
    r = FastPlateReader(rec).read(crop())
    assert r.text == "ABC1D23"
    assert abs(r.confidence - np.mean([0.9, 0.8, 1.0, 0.7, 0.9, 0.9, 0.9])) < 1e-9
    assert rec.last.ndim == 2  # modelo em cinza


def test_modo_rgb_mantem_tres_canais():
    rec = FakeRec("ABC1234", np.ones(7), mode="rgb")
    FastPlateReader(rec).read(crop())
    assert rec.last.shape[2] == 3


def test_texto_vazio_ou_crop_invalido_da_none():
    assert FastPlateReader(FakeRec("___", np.zeros(3))).read(crop()) is None
    assert FastPlateReader(FakeRec("ABC", np.ones(3))).read(np.zeros((1, 1, 3), np.uint8)) is None


def test_nao_corrige_caracteres_e_confianca_zero_sem_probs():
    r = FastPlateReader(FakeRec("0I0", None)).read(crop())
    assert r.text == "0I0" and r.confidence == 0.0
