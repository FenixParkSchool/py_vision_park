import numpy as np
import pytest

from app.recognition.preprocessing import (
    PLATE_SIZE,
    guess_vehicle_type,
    prepare_plate,
    stack_two_lines,
)


def placa_de_duas_linhas():
    img = np.zeros((40, 20, 3), dtype=np.uint8)  # altura 40, largura 20
    img[20:] = 255  # metade de cima preta, metade de baixo branca
    return img


def test_stack_poe_a_linha_de_cima_a_esquerda():
    junto = stack_two_lines(placa_de_duas_linhas())
    assert junto.shape == (20, 40, 3)
    assert junto[:, :20].max() == 0    # esquerda = linha de cima
    assert junto[:, 20:].min() == 255  # direita = linha de baixo


def test_stack_com_altura_impar_nao_quebra():
    assert stack_two_lines(np.zeros((41, 20, 3), dtype=np.uint8)).shape == (20, 40, 3)


def test_stack_com_altura_1_levanta_erro():
    with pytest.raises(ValueError):
        stack_two_lines(np.zeros((1, 20, 3), dtype=np.uint8))


@pytest.mark.parametrize("tipo, forma", [("car", (43, 127, 3)), ("motorcycle", (85, 100, 3))])
def test_saida_tem_sempre_o_tamanho_fixo(tipo, forma):
    saida = prepare_plate(np.zeros(forma, dtype=np.uint8), tipo)
    assert saida.shape == (PLATE_SIZE[1], PLATE_SIZE[0], 3)


def test_tipo_desconhecido_levanta_erro():
    with pytest.raises(ValueError):
        prepare_plate(np.zeros((40, 120, 3), dtype=np.uint8), "caminhao")


def test_recorte_vazio_levanta_erro():
    with pytest.raises(ValueError):
        prepare_plate(np.zeros((0, 0, 3), dtype=np.uint8), "car")


def test_palpite_por_proporcao():
    assert guess_vehicle_type(np.zeros((43, 127, 3), dtype=np.uint8)) == "car"
    assert guess_vehicle_type(np.zeros((85, 100, 3), dtype=np.uint8)) == "motorcycle"