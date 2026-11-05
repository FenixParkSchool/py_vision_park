"""Treina o detector de placas (mapa de probabilidade) no RodoSol-ALPR. CPU, poucos minutos.

Uso:  .venv\\Scripts\\python.exe -m training.train_detector --per-folder 600 --epochs 12
Saída: models/detector.keras
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from app.recognition.tf_models import DETECTOR_INPUT_WIDTH, DETECTOR_PATH, DETECTOR_STRIDE, to_gray
from training.rds import find_images, load_sample
from training.split import split_paths

RAIZ = Path("data/rodosol")
SEED = 0
POS_WEIGHT = 10.0  # células de placa são ~0,5% do mapa; pesam mais na perda
MAX_ZOOM = 2.0  # ampliação máxima no aumento de dados (placas maiores, como de perto na webcam)


def pick(paths: list[Path], per_folder: int) -> list[Path]:
    """Pega `per_folder` imagens espaçadas de cada pasta (ordem estável)."""
    por_pasta: dict[Path, list[Path]] = {}
    for p in paths:
        por_pasta.setdefault(p.parent, []).append(p)
    saida = []
    for lista in por_pasta.values():
        passo = max(1, len(lista) // per_folder)
        saida += lista[::passo][:per_folder]
    return saida


def make_sample(gray: np.ndarray, box, zoom: float, rng: np.random.Generator):
    """Redimensiona (escala base x zoom), recorta uma janela do tamanho da entrada e monta a máscara."""
    altura, largura = gray.shape
    escala = DETECTOR_INPUT_WIDTH / largura * zoom
    img = cv2.resize(gray, (round(largura * escala), round(altura * escala)), interpolation=cv2.INTER_AREA)
    in_w, in_h = DETECTOR_INPUT_WIDTH, round(altura * DETECTOR_INPUT_WIDTH / largura)
    bx1, by1 = box.x * escala, box.y * escala
    bx2, by2 = (box.x + box.width) * escala, (box.y + box.height) * escala

    def origem(lo, hi, janela, total):
        # janela que contém a placa inteira, se der; senão qualquer uma
        a, b = max(0, int(np.ceil(hi)) - janela), min(int(lo), total - janela)
        return int(rng.integers(a, b + 1)) if a <= b else int(rng.integers(0, total - janela + 1))

    x0 = origem(bx1, bx2, in_w, img.shape[1])
    y0 = origem(by1, by2, in_h, img.shape[0])
    janela = img[y0 : y0 + in_h, x0 : x0 + in_w]

    # Marca as células cujo centro cai dentro da placa (arredondar, e não floor/ceil,
    # deixa a caixa decodificada justa; caixa folgada estraga o palpite carro/moto).
    mascara = np.zeros((in_h // DETECTOR_STRIDE, in_w // DETECTOR_STRIDE), np.float32)
    s = DETECTOR_STRIDE
    c1, r1 = round((bx1 - x0) / s), round((by1 - y0) / s)
    c2, r2 = max(c1 + 1, round((bx2 - x0) / s)), max(r1 + 1, round((by2 - y0) / s))
    mascara[max(0, r1) : max(0, r2), max(0, c1) : max(0, c2)] = 1.0
    return janela, mascara


def load_set(paths: list[Path], rng: np.random.Generator, augment: bool):
    xs, ys = [], []
    for caminho in paths:
        box = load_sample(caminho).annotation.bbox
        gray = to_gray(cv2.imread(str(caminho)))
        zooms = [1.0, float(rng.uniform(1.0, MAX_ZOOM))] if augment else [1.0]
        for z in zooms:
            x, y = make_sample(gray, box, z, rng)
            xs.append(x)
            ys.append(y)
    return np.stack(xs)[..., None], np.stack(ys)[..., None]


def build_model():
    import keras
    from keras import layers

    def bloco(x, filtros, passo=1, dil=1):
        x = layers.Conv2D(filtros, 3, strides=passo, dilation_rate=dil, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        return layers.ReLU()(x)

    entrada = keras.Input((None, None, 1))
    x = bloco(entrada, 16, 2)
    x = bloco(x, 16)
    x = bloco(x, 32, 2)
    s4 = bloco(x, 32)  # 1/4
    x = bloco(s4, 64, 2)  # 1/8
    x = bloco(x, 64)
    x = bloco(x, 64, dil=2)
    x = layers.UpSampling2D(2)(x)  # volta a 1/4
    x = layers.Concatenate()([x, s4])
    x = bloco(x, 32)
    saida = layers.Conv2D(1, 1, activation="sigmoid")(x)
    return keras.Model(entrada, saida)


def weighted_bce(y, p):
    from keras import ops

    p = ops.clip(p, 1e-6, 1 - 1e-6)
    bce = -(y * ops.log(p) + (1 - y) * ops.log(1 - p))
    return ops.mean((1 + (POS_WEIGHT - 1) * y) * bce)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-folder", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--out", type=Path, default=DETECTOR_PATH)
    args = ap.parse_args()

    import keras
    import tensorflow as tf

    keras.utils.set_random_seed(SEED)
    rng = np.random.default_rng(SEED)
    divisao = split_paths(find_images(RAIZ))
    x_tr, y_tr = load_set(pick(divisao["train"], args.per_folder), rng, augment=True)
    x_va, y_va = load_set(pick(divisao["val"], 50), rng, augment=False)
    print("treino", x_tr.shape, "validação", x_va.shape)

    def preparar(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        x = tf.image.random_brightness(x, 0.15)
        x = tf.image.random_contrast(x, 0.7, 1.3)
        return tf.clip_by_value(x, 0, 1), y

    treino = (
        tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
        .shuffle(len(x_tr), seed=SEED)
        .batch(16)
        .map(preparar)
        .prefetch(2)
    )
    validacao = (x_va.astype(np.float32) / 255.0, y_va)

    modelo = build_model()
    modelo.compile(optimizer=keras.optimizers.Adam(2e-3), loss=weighted_bce)
    modelo.fit(
        treino,
        validation_data=validacao,
        epochs=args.epochs,
        callbacks=[keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5)],
        verbose=2,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    modelo.save(args.out)
    print("Salvo em", args.out)


if __name__ == "__main__":
    main()
