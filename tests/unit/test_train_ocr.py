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


def test_margin_crop_caixa_relativa():
    from app.schemas.recognition import BoundingBox
    from training.train_ocr import margin_crop

    img = np.zeros((200, 300, 3), np.uint8)
    img[100:120, 100:160] = 255  # a "placa"
    crop, rel = margin_crop(img, BoundingBox(x=100, y=100, width=60, height=20), 0.3)
    assert crop.shape == (20 + 2 * 6, 60 + 2 * 18, 3)
    assert (rel.x, rel.y, rel.width, rel.height) == (18, 6, 60, 20)
    assert (crop[rel.y : rel.y + rel.height, rel.x : rel.x + rel.width] == 255).all()
    # na borda do frame a margem é cortada e a caixa relativa continua certa
    crop, rel = margin_crop(img, BoundingBox(x=0, y=0, width=50, height=20), 0.3)
    assert (rel.x, rel.y) == (0, 0) and crop.shape[:2] == (26, 65)


def _fake_cropset():
    from training.train_ocr import CropSet

    c1 = np.full((26, 96, 3), 100, np.uint8)
    c2 = np.full((30, 90, 3), 50, np.uint8)
    return CropSet(
        {
            "flat": np.concatenate([c1.ravel(), c2.ravel()]),
            "shapes": np.array([c1.shape, c2.shape], np.int32),
            "boxes": np.array([[18, 6, 60, 20], [10, 5, 60, 20]], np.int32),
            "labels": np.array([[1] * 7, [2] * 7], np.int32),
            "moto": np.array([False, True]),
        }
    )


def test_cache_roundtrip_e_chave(tmp_path, monkeypatch):
    import training.train_ocr as t

    chamadas = []

    def fake_build(paths):
        chamadas.append(1)
        cs = _fake_cropset()
        return {"flat": cs.flat, "shapes": cs.shapes, "boxes": cs.boxes, "labels": cs.labels, "moto": cs.moto}

    monkeypatch.setattr(t, "build_cache", fake_build)
    a = t.load_cached("k", lambda: [], tmp_path)
    b = t.load_cached("k", lambda: [], tmp_path)  # 2a vez: lê o .npz
    assert chamadas == [1] and (tmp_path / "k.npz").is_file()
    assert (a.crop(1) == b.crop(1)).all() and b.vehicle_type(1) == "motorcycle" and b.box(0).x == 18
    assert t.cache_name("train", 50) != t.cache_name("train", 3000) != t.cache_name("val", 3000)


def test_train_image_varia_a_cada_chamada_e_augment_por_imagem():
    from training.train_ocr import augment, train_image

    cs = _fake_cropset()
    rng = np.random.default_rng(0)
    a, b = train_image(cs, 0, rng), train_image(cs, 0, rng)
    assert a.shape == (48, 128, 1) and a.dtype == np.float32 and not np.array_equal(a, b)
    base = np.full((48, 128, 1), 0.5, np.float32)
    fatores = {round(float(augment(base, rng).mean()), 3) for _ in range(5)}
    assert len(fatores) > 1 and all(0 <= f <= 1 for f in fatores)  # cada imagem sorteia o seu


def test_checkpoint_salva_so_quando_val_loss_melhora():
    from training.train_ocr import checkpoint_callback

    salvos = []
    cb = checkpoint_callback(lambda e, logs: salvos.append(e))
    for e, v in enumerate([1.0, 0.8, 0.9, 0.7, 0.7]):
        cb.on_epoch_end(e, {"val_loss": v})
    assert salvos == [0, 1, 3]


def test_normalizacao_dentro_do_modelo_e_carrega_sem_custom_objects(tmp_path):
    import keras

    from training.train_ocr import build_positions

    modelo, _ = build_positions()
    x = np.random.default_rng(0).random((2, 48, 128, 1)).astype(np.float32)
    caminho = tmp_path / "m.keras"
    modelo.save(caminho)
    carregado = keras.saving.load_model(caminho, compile=False)
    esperado = modelo.predict(x, verbose=0)
    assert np.allclose(carregado.predict(x, verbose=0), esperado, atol=1e-5)
    # brilho/contraste globais quase não mudam a saída (padronização por imagem)
    assert np.allclose(modelo.predict(0.5 * x + 0.2, verbose=0), esperado, atol=2e-2)
