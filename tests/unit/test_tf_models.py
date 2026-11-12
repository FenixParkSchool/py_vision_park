import subprocess
import sys

import numpy as np
import pytest

from app.recognition.tf_models import (
    ALPHABET,
    PLATE_LENGTH,
    TfPlateDetector,
    TfPlateReader,
    decode_ctc,
    decode_positions,
    detector_input,
    mask_to_detections,
    ocr_input,
)


def one_hot(texto: str, p: float = 0.9) -> np.ndarray:
    probs = np.full((PLATE_LENGTH, len(ALPHABET)), (1 - p) / (len(ALPHABET) - 1), np.float32)
    for i, c in enumerate(texto):
        probs[i, ALPHABET.index(c)] = p
    return probs


def test_detector_input_reduz_normaliza_e_completa_ate_multiplo_de_8():
    frame = np.full((720, 1280, 3), 255, np.uint8)
    x, escala = detector_input(frame, 384)
    assert x.shape == (1, 216, 384, 1) and x.dtype == np.float32
    assert escala == pytest.approx(0.3)
    assert x.max() == pytest.approx(1.0)
    x, _ = detector_input(np.zeros((100, 100, 3), np.uint8), 100)
    assert x.shape == (1, 104, 104, 1)  # 100 -> 104 com zeros


def test_mask_to_detections_converte_celulas_para_pixels_do_frame():
    prob = np.zeros((54, 96), np.float32)
    prob[10:13, 20:30] = 0.8  # 3 x 10 células
    prob[40, 5] = 0.9  # célula isolada: ruído, descartada (min_cells=2)
    [d] = mask_to_detections(prob, scale=0.3, frame_shape=(720, 1280, 3))
    fator = 4 / 0.3
    assert d.bbox.x == int(20 * fator) and d.bbox.y == int(10 * fator)
    assert d.bbox.width == pytest.approx(10 * fator, abs=1)
    assert d.bbox.height == pytest.approx(3 * fator, abs=1)
    assert d.confidence == pytest.approx(0.8)


def test_mask_to_detections_limita_a_borda_do_frame():
    prob = np.zeros((10, 10), np.float32)
    prob[8:10, 8:10] = 0.7
    [d] = mask_to_detections(prob, scale=1.0, frame_shape=(38, 38))
    assert d.bbox.x + d.bbox.width <= 38 and d.bbox.y + d.bbox.height <= 38


def test_mask_vazia_nao_detecta_nada():
    assert mask_to_detections(np.zeros((10, 10), np.float32), 1.0, (40, 40)) == []


def test_detector_com_modelo_falso():
    recebidos = []

    def prever(x):
        recebidos.append(x.shape)
        prob = np.zeros((1, x.shape[1] // 4, x.shape[2] // 4, 1), np.float32)
        prob[0, 5:7, 10:20, 0] = 0.95
        return prob

    detector = TfPlateDetector(prever, input_width=384)
    [d] = detector.detect(np.zeros((480, 640, 3), np.uint8))
    assert recebidos == [(1, 288, 384, 1)]
    assert d.confidence == pytest.approx(0.95)
    assert d.bbox.x == int(10 * 4 / 0.6)


def test_decode_positions_devolve_texto_cru_e_menor_confianca():
    probs = one_hot("ABC1D23")
    probs[3] = one_hot("1111111", p=0.55)[0]
    leitura = decode_positions(probs)
    assert leitura.text == "ABC1D23"
    assert leitura.confidence == pytest.approx(0.55)


def test_decode_positions_nao_corrige_ambiguidade():
    assert decode_positions(one_hot("ABCID23")).text == "ABCID23"


def test_decode_positions_formato_errado_levanta_erro():
    with pytest.raises(ValueError):
        decode_positions(np.zeros((6, len(ALPHABET))))


def test_ocr_input_carro_e_moto_tem_mesmo_formato():
    carro = np.zeros((45, 130, 3), np.uint8)
    moto = np.zeros((50, 57, 3), np.uint8)
    assert ocr_input(carro).shape == (1, 48, 128, 1)
    assert ocr_input(moto).shape == (1, 48, 128, 1)


def test_reader_com_modelo_falso():
    entradas = []

    def prever(x):
        entradas.append(x.shape)
        return one_hot("ABC1234")[None]

    leitor = TfPlateReader(prever)
    leitura = leitor.read(np.zeros((45, 130, 3), np.uint8))
    assert leitura.text == "ABC1234" and leitura.confidence == pytest.approx(0.9)
    assert entradas == [(1, 48, 128, 1)]


def test_reader_recorte_minusculo_devolve_none():
    leitor = TfPlateReader(lambda x: pytest.fail("não deveria chamar o modelo"))
    assert leitor.read(np.zeros((1, 50, 3), np.uint8)) is None


def test_from_file_sem_modelo_levanta_erro_claro(tmp_path):
    with pytest.raises(FileNotFoundError):
        TfPlateDetector.from_file(tmp_path / "nao_existe.keras")
    with pytest.raises(FileNotFoundError):
        TfPlateReader.from_file(tmp_path / "nao_existe.keras")


def test_importar_modulo_nao_carrega_tensorflow():
    codigo = "import sys, app.recognition.tf_models; print('tensorflow' in sys.modules)"
    saida = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, check=True)
    assert saida.stdout.strip() == "False"


def ctc_probs(passos: list[tuple[int | None, float]]) -> np.ndarray:
    """Cada passo = (índice do caractere ou None para blank, probabilidade)."""
    n = len(ALPHABET) + 1
    saida = np.full((len(passos), n), 0.0, np.float32)
    for t, (i, p) in enumerate(passos):
        i = n - 1 if i is None else i
        saida[t, :] = (1 - p) / (n - 1)
        saida[t, i] = p
    return saida


def test_decode_ctc_colapsa_repetidos_e_remove_blank():
    a, b = ALPHABET.index("A"), ALPHABET.index("B")
    leitura = decode_ctc(ctc_probs([(a, 0.9), (a, 0.8), (None, 0.99), (a, 0.7), (b, 0.95), (None, 0.9)]))
    assert leitura.text == "AAB"  # blank separa A repetido
    assert leitura.confidence == pytest.approx(0.7)


def test_decode_ctc_confianca_e_minimo_dos_passos_emitidos():
    a, b = ALPHABET.index("A"), ALPHABET.index("B")
    leitura = decode_ctc(ctc_probs([(a, 0.9), (None, 0.3), (b, 0.6)]))
    assert leitura.text == "AB"
    assert leitura.confidence == pytest.approx(0.6)


def test_decode_ctc_so_blank_devolve_none():
    assert decode_ctc(ctc_probs([(None, 0.9)] * 5)) is None


def test_decode_ctc_formato_errado_levanta_erro():
    with pytest.raises(ValueError):
        decode_ctc(np.zeros((8, len(ALPHABET))))


def test_ocr_input_com_tipo_explicito_ignora_o_palpite():
    largo = np.zeros((20, 130, 3), np.uint8)
    largo[:10] = 255  # metade de cima branca
    como_carro = ocr_input(largo)
    como_moto = ocr_input(largo, "motorcycle")
    assert not np.array_equal(como_carro, como_moto)
    assert np.array_equal(como_carro, ocr_input(largo, "car"))


def test_reader_detecta_saida_ctc():
    a = ALPHABET.index("A")
    leitor = TfPlateReader(lambda x: ctc_probs([(a, 0.9), (None, 0.9)] + [(None, 0.9)] * 30)[None])
    assert leitor.read(np.zeros((40, 120, 3), np.uint8)).text == "A"
