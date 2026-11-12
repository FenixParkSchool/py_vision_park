"""Treina o OCR com recortes do RodoSol-ALPR. CPU.

Arquiteturas: `ctc` (CRNN convolucional, saída (T, 37), padrão) e `positions` (7 x 36).

Uso:  .venv\\Scripts\\python.exe -m training.train_ocr --arch ctc --per-folder 3000 --epochs 30
Saída: models/ocr.keras (modelo de inferência) e a ficha .json
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from app.recognition.model_card import write_card
from app.recognition.preprocessing import crop_region
from app.recognition.tf_models import ALPHABET, OCR_PATH, PLATE_LENGTH, decode_ctc, ocr_input
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
            xs.append(ocr_input(recorte, a.vehicle_type)[0])
            ys.append(rotulo)
    return np.stack(xs), np.stack(ys)


def build_positions():
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
    modelo = keras.Model(entrada, saida)
    return modelo, modelo


def build_ctc():
    """CRNN só convolucional. Devolve (modelo de treino com logits, modelo de inferência com softmax).

    Altura 48 -> 24 -> 12 -> 6 -> 3 -> 1; largura 128 -> 64 -> 32 (32 passos de tempo).
    A última classe (índice 36) é o blank do CTC.
    """
    import keras
    from keras import layers

    def conv(x, filtros):
        x = layers.Conv2D(filtros, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        return layers.ReLU()(x)

    entrada = keras.Input((48, 128, 1))
    x = conv(entrada, 32)
    x = layers.MaxPooling2D((2, 2))(x)  # 24x64
    x = conv(x, 64)
    x = layers.MaxPooling2D((2, 2))(x)  # 12x32
    x = conv(conv(x, 128), 128)
    x = layers.MaxPooling2D((2, 1))(x)  # 6x32
    x = conv(conv(x, 128), 128)
    x = layers.MaxPooling2D((2, 1))(x)  # 3x32
    x = layers.Conv2D(192, (3, 1), padding="valid", use_bias=False)(x)  # 1x32: a altura vira canais
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Reshape((32, 192))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv1D(192, 3, padding="same", activation="relu")(x)
    logits = layers.Dense(len(ALPHABET) + 1)(x)
    return keras.Model(entrada, logits), keras.Model(entrada, layers.Softmax()(logits))


def ctc_loss(y_true, logits):
    """Perda CTC com rótulos de comprimento fixo 7; o blank é a última classe."""
    from keras import ops

    lote, passos = ops.shape(logits)[0], logits.shape[1]
    return ops.ctc_loss(
        y_true,
        logits,
        ops.full((lote,), PLATE_LENGTH, "int32"),
        ops.full((lote,), passos, "int32"),
        mask_index=len(ALPHABET),
    )


def decode_batch(probs: np.ndarray, arch: str) -> list[str]:
    if arch == "positions":
        return ["".join(ALPHABET[i] for i in p.argmax(axis=1)) for p in probs]
    leituras = [decode_ctc(p) for p in probs]
    return [x.text if x else "" for x in leituras]


def val_scores(probs: np.ndarray, y: np.ndarray, arch: str) -> tuple[float, float]:
    """(acerto por caractere, acerto de placa inteira) com decodificação gulosa."""
    verdade = ["".join(ALPHABET[i] for i in r) for r in y]
    lidas = decode_batch(probs, arch)
    chars = sum(sum(a == b for a, b in zip(p, t)) for p, t in zip(lidas, verdade))
    inteiras = sum(p == t for p, t in zip(lidas, verdade))
    return chars / (len(verdade) * PLATE_LENGTH), inteiras / len(verdade)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["positions", "ctc"], default="ctc")
    ap.add_argument("--per-folder", type=int, default=1500)
    ap.add_argument("--copies", type=int, default=2, help="recortes com caixa tremida por imagem")
    ap.add_argument("--epochs", type=int, default=15, help="máximo; a parada antecipada pode acabar antes")
    ap.add_argument("--out", type=Path, default=OCR_PATH)
    args = ap.parse_args()

    import keras
    import tensorflow as tf

    keras.utils.set_random_seed(SEED)
    rng = np.random.default_rng(SEED)
    divisao = split_paths(find_images(RAIZ))
    x_tr, y_tr = load_set(pick(divisao["train"], args.per_folder), rng, copies=args.copies)
    x_va, y_va = load_set(pick(divisao["val"], 100), rng, copies=1)
    print("treino", x_tr.shape, "validação", x_va.shape, flush=True)

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
    if args.arch == "ctc":
        treinavel, inferencia = build_ctc()
        treinavel.compile(optimizer=keras.optimizers.Adam(1e-3), loss=ctc_loss)
    else:
        treinavel, inferencia = build_positions()
        treinavel.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    class ValScores(keras.callbacks.Callback):
        """Acerto por caractere e de placa inteira na validação (decodificação gulosa)."""

        def on_epoch_end(self, epoch, logs=None):
            probs = inferencia.predict(x_va, batch_size=256, verbose=0)
            logs["val_char_accuracy"], logs["val_plate_accuracy"] = val_scores(probs, y_va, args.arch)
            print(
                f"  época {epoch + 1}: val_char_accuracy={logs['val_char_accuracy']:.4f} "
                f"val_plate_accuracy={logs['val_plate_accuracy']:.4f}",
                flush=True,
            )

    historico = treinavel.fit(
        treino,
        validation_data=(x_va, y_va),
        epochs=args.epochs,
        callbacks=[
            ValScores(),
            keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ],
        verbose=2,
    )
    h = historico.history
    melhor = int(np.argmin(h["val_loss"]))  # a mesma época que o EarlyStopping restaura
    args.out.parent.mkdir(parents=True, exist_ok=True)
    inferencia.save(args.out)
    write_card(
        args.out,
        "ocr",
        params={
            "arch": args.arch,
            "per_folder": args.per_folder,
            "copies": args.copies,
            "epochs_max": args.epochs,
            "epochs_rodadas": len(h["val_loss"]),
            "seed": SEED,
        },
        metrics={
            "best_epoch": melhor + 1,
            **{f"best_{k}": float(v[melhor]) for k, v in h.items() if k.startswith(("val_", "loss"))},
        },
        dataset={"train": len(x_tr), "val": len(x_va)},
    )
    print("Salvo em", args.out, "(ficha:", args.out.with_suffix(".json"), ") melhor época", melhor + 1)


if __name__ == "__main__":
    main()
