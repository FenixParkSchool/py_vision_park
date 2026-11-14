import csv
from pathlib import Path

import numpy as np
import pytest

from app.schemas.recognition import BoundingBox
from training import exportar_fpo
from training.exportar_fpo import expand_box, nome_saida


def test_expand_box_aumenta_a_margem_de_cada_lado():
    caixa = expand_box(BoundingBox(x=100, y=50, width=100, height=40), 0.1, 1280, 720)
    assert (caixa.x, caixa.y, caixa.width, caixa.height) == (90, 46, 120, 48)


def test_expand_box_nao_passa_da_imagem():
    caixa = expand_box(BoundingBox(x=2, y=1, width=100, height=40), 0.5, 120, 50)
    assert caixa.x == 0 and caixa.y == 0
    assert caixa.x + caixa.width <= 120 and caixa.y + caixa.height <= 50


def test_nome_saida_usa_pasta_e_imagem():
    assert nome_saida(Path("data/rodosol/cars-me/img_000123.jpg")) == "cars-me_img_000123.png"


def test_exportar_grava_png_e_csv_no_formato_do_fast_plate_ocr(tmp_path, monkeypatch):
    frame = np.full((720, 1280, 3), 200, np.uint8)

    class Amostra:
        class annotation:
            plate = "ABC1D23"
            bbox = BoundingBox(x=100, y=100, width=120, height=40)

    monkeypatch.setattr(exportar_fpo, "load_sample", lambda p: Amostra)
    monkeypatch.setattr(exportar_fpo.cv2, "imread", lambda p: frame)
    pasta = tmp_path / "train"

    n = exportar_fpo.exportar([Path("cars-me/img_1.jpg")], pasta, 0.08)

    assert n == 1
    linhas = list(csv.reader(open(tmp_path / "train.csv", encoding="utf-8")))
    assert linhas == [["image_path", "plate_text"], ["train/cars-me_img_1.png", "ABC1D23"]]
    assert (pasta / "cars-me_img_1.png").exists()


def test_exportar_com_regiao_grava_a_coluna_plate_region(tmp_path, monkeypatch):
    frame = np.full((720, 1280, 3), 200, np.uint8)

    class Amostra:
        class annotation:
            plate = "ABC1234"
            bbox = BoundingBox(x=100, y=100, width=120, height=40)

    monkeypatch.setattr(exportar_fpo, "load_sample", lambda p: Amostra)
    monkeypatch.setattr(exportar_fpo.cv2, "imread", lambda p: frame)

    exportar_fpo.exportar([Path("cars-br/img_2.jpg")], tmp_path / "val", 0.08, regiao="Brazil")

    linhas = list(csv.reader(open(tmp_path / "val.csv", encoding="utf-8")))
    assert linhas == [["image_path", "plate_text", "plate_region"], ["val/cars-br_img_2.png", "ABC1234", "Brazil"]]


@pytest.mark.parametrize("margem", [0.0, 0.08])
def test_expand_box_margem_zero_mantem_a_caixa(margem):
    caixa = BoundingBox(x=10, y=10, width=50, height=20)
    nova = expand_box(caixa, margem, 1280, 720)
    assert nova.width >= caixa.width and nova.height >= caixa.height
