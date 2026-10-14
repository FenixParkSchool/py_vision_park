import numpy as np

from app.recognition.preprocessing import crop_region
from app.schemas.recognition import BoundingBox


def frame_vazio():
    return np.zeros((100, 200, 3), dtype=np.uint8)  # altura 100, largura 200


def test_recorte_dentro_do_frame_tem_o_tamanho_da_caixa():
    recorte = crop_region(frame_vazio(), BoundingBox(x=10, y=20, width=30, height=40))
    assert recorte.shape == (40, 30, 3)


def test_recorte_pega_a_regiao_certa():
    frame = frame_vazio()
    frame[20:60, 10:40] = 255  # a região exata da caixa
    recorte = crop_region(frame, BoundingBox(x=10, y=20, width=30, height=40))
    assert recorte.min() == 255


def test_caixa_que_passa_da_borda_e_cortada():
    recorte = crop_region(frame_vazio(), BoundingBox(x=190, y=90, width=50, height=50))
    assert recorte.shape == (10, 10, 3)


def test_caixa_totalmente_fora_devolve_none():
    assert crop_region(frame_vazio(), BoundingBox(x=300, y=10, width=10, height=10)) is None