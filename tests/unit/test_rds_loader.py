import cv2
import numpy as np
import pytest

from training.rds import find_images, load_sample

ANOTACAO = "type: car\nplate: ABC1D23\nlayout: Mercosur\ncorners: 10,10 50,10 50,30 10,30\n"


def criar_par(pasta, nome, anotacao=ANOTACAO):
    pasta.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(pasta / f"{nome}.jpg"), np.zeros((40, 80, 3), dtype=np.uint8))
    if anotacao is not None:
        (pasta / f"{nome}.txt").write_text(anotacao, encoding="utf-8")


def test_encontra_imagens_em_subpastas_em_ordem_estavel(tmp_path):
    criar_par(tmp_path / "b", "img_2")
    criar_par(tmp_path / "a", "img_1")
    assert [p.name for p in find_images(tmp_path)] == ["img_1.jpg", "img_2.jpg"]


def test_carrega_imagem_e_gabarito(tmp_path):
    criar_par(tmp_path, "img_1")
    amostra = load_sample(tmp_path / "img_1.jpg")
    assert amostra.image_path == tmp_path / "img_1.jpg"
    assert amostra.annotation.plate == "ABC1D23"


def test_sem_anotacao_levanta_erro(tmp_path):
    criar_par(tmp_path, "img_1", anotacao=None)
    with pytest.raises(FileNotFoundError):
        load_sample(tmp_path / "img_1.jpg")


def test_anotacao_invalida_informa_qual_arquivo(tmp_path):
    criar_par(tmp_path, "img_1", anotacao="type: car\n")
    with pytest.raises(ValueError, match="img_1.txt"):
        load_sample(tmp_path / "img_1.jpg")