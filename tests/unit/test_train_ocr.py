import numpy as np

from app.recognition.tf_models import ALPHABET
from training.train_ocr import build_ctc, ctc_loss, val_scores


def _ctc_probs(texto: str, passos: int = 24) -> np.ndarray:
    """Cada caractere em 2 passos seguidos, separados por blank; sobra blank no fim."""
    p = np.full((passos, len(ALPHABET) + 1), 0.001, np.float32)
    t = 0
    for c in texto:
        p[t, ALPHABET.index(c)] = p[t + 1, ALPHABET.index(c)] = 0.9
        p[t + 2, -1] = 0.9
        t += 3
    p[t:, -1] = 0.9
    return p


def test_val_scores_ctc():
    y = np.array([[ALPHABET.index(c) for c in "ABC1D23"]] * 2)
    certa = _ctc_probs("ABC1D23")
    errada = _ctc_probs("ABC1D24")
    char, inteira = val_scores(np.stack([certa, errada]), y, "ctc")
    assert inteira == 0.5
    assert char == (7 + 6) / 14


def test_build_ctc_saidas_e_perda_finita():
    treino, inferencia = build_ctc()
    x = np.random.rand(2, 48, 128, 1).astype(np.float32)
    assert inferencia.predict(x, verbose=0).shape == (2, 32, len(ALPHABET) + 1)
    y = np.array([[1, 2, 3, 4, 5, 6, 7]] * 2, np.int32)
    perda = np.asarray(ctc_loss(y, treino(x)))
    assert np.isfinite(perda).all()
