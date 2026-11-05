"""Treina o OCR (7 posições x 36 classes) com recortes do RodoSol-ALPR. CPU, poucos minutos.

Uso:  .venv\\Scripts\\python.exe -m training.train_ocr --per-folder 1500 --epochs 15
Saída: models/ocr.keras
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from app.recognition.preprocessing import crop_region
from app.recognition.tf_models import ALPHABET, OCR_PATH, PLATE_LENGTH, ocr_input
from app.schemas.recognition import BoundingBox
from training.rds import find_images, load_sample
from training.split import split_paths
from training.train_detector import RAIZ, SEED, pick


def jitter_box(box: BoundingBox, shape, rng: np.random.Generator) -> BoundingBox:
    """Desloca as bordas da caixa como um detector imperfeito faria (mais folga que aperto)."""
    altura, largura = shape[:2]
    dx = rng.uniform(-0.05, 0.12, 2) * box.width
    dy = rng.uniform(-0.05, 0.25, 2) * box.height
    x1 = max(0, int(box.x - dx[0]))
    y1 = max(0, int(box.y - dy[0]))
    x2 = min(largura, int(box.x + box.width + dx[1]))
    y2 = min(altura, int(box.y + box.height + dy[1]))
    return BoundingBox(x=x1, y=y1, width=max(1, x2 - x1), height=max(1, y2 - y1))


def encode(text: str) -> np.ndarray | None:
    if len(text) != PLATE_LENGTH or any(c not in ALPHABET for c in text):
        return None
    return np.array([ALPHABET.index(c) for c in text], np.int32)


def load_set(paths: list[Path], rng: np.random.Generator, copies: int):
    """copies=0: só a caixa do gabarito; copies>0: essa quantidade de caixas tremidas por imagem."""
    xs, ys = [], []
    for caminho in paths:
        a = load_sample(caminho).annotation
        rotulo = encode(a.plate)
        if rotulo is None:
            continue
        img = cv2.imread(str(caminho))
        caixas = [jitter_box(a.bbox, img.shape, rng) for _ in range(copies)] or [a.bbox]
        for caixa in caixas:
            recorte = crop_region(img, caixa)
            if recorte is None or min(recorte.shape[:2]) < 2:
                continue
            xs.append(ocr_input(recorte)[0])
            ys.append(rotulo)
    return np.stack(xs), np.stack(ys)


def build_model():
    import keras
    from keras import layers

    def bloco(x, filtros):
        for _ in range(2):
            x = layers.Conv2D(filtros, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        return layers.MaxPooling2D(2)(x)

    entrada = keras.Input((48, 128, 1))
    x = bloco(entrada, 32)  # 24x64
    x = bloco(x, 64)  # 12x32
    x = bloco(x, 128)  # 6x16
    x = bloco(x, 128)  # 3x8
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(PLATE_LENGTH * len(ALPHABET))(x)
    x = layers.Reshape((PLATE_LENGTH, len(ALPHABET)))(x)
    saida = layers.Softmax()(x)
    return keras.Model(entrada, saida)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-folder", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--out", type=Path, default=OCR_PATH)
    args = ap.parse_args()

    import keras
    import tensorflow as tf

    keras.utils.set_random_seed(SEED)
    rng = np.random.default_rng(SEED)
    divisao = split_paths(find_images(RAIZ))
    x_tr, y_tr = load_set(pick(divisao["train"], args.per_folder), rng, copies=2)
    x_va, y_va = load_set(pick(divisao["val"], 100), rng, copies=1)
    print("treino", x_tr.shape, "validação", x_va.shape)

    def preparar(x, y):
        x = tf.image.random_brightness(x, 0.15)
        x = tf.image.random_contrast(x, 0.7, 1.3)
        return tf.clip_by_value(x, 0, 1), y

    treino = (
        tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
        .shuffle(len(x_tr), seed=SEED)
        .batch(64)
        .map(preparar)
        .prefetch(2)
    )
    modelo = build_model()
    modelo.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],  # acerto por caractere
    )
    modelo.fit(
        treino,
        validation_data=(x_va, y_va),
        epochs=args.epochs,
        callbacks=[keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5)],
        verbose=2,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    modelo.save(args.out)
    print("Salvo em", args.out)


if __name__ == "__main__":
    main()
