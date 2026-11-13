"""Treina o OCR com recortes do RodoSol-ALPR. CPU.

Arquiteturas: `ctc` (CRNN convolucional, saída (T, 37), padrão) e `positions` (7 x 36).

Uso:  .venv\\Scripts\\python.exe -m training.train_ocr --arch positions --per-folder 3000 --epochs 30
Saída: models/ocr.keras (modelo de inferência) e a ficha .json

- Cache (data/cache/): recortes com margem de cada imagem, lidos do disco só na 1a execução.
- A cada época o tremor da caixa e a augmentação são sorteados de novo, imagem por imagem.
- A validação usa todas as imagens de validação com a caixa do gabarito (sem tremor).
- O melhor modelo (menor val_loss) é salvo em `--out` (e a ficha atualizada) a cada melhora.
- A normalização de contraste por imagem é a primeira camada dos dois modelos (`ocr_input` não muda).
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

CACHE_DIR = Path("data/cache")
MARGIN = 0.30  # folga de cada lado da caixa do gabarito no recorte em cache
BATCH = 64


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


def margin_crop(img: np.ndarray, box: BoundingBox, margin: float = MARGIN):
    """Recorte com margem (caixa expandida `margin` por lado) e a caixa do gabarito relativa a ele."""
    altura, largura = img.shape[:2]
    mx, my = int(box.width * margin), int(box.height * margin)
    x1, y1 = max(0, box.x - mx), max(0, box.y - my)
    x2, y2 = min(largura, box.x + box.width + mx), min(altura, box.y + box.height + my)
    if x2 <= x1 or y2 <= y1:
        return None
    rel = BoundingBox(
        x=max(0, box.x - x1),
        y=max(0, box.y - y1),
        width=max(1, min(box.x + box.width, x2) - max(box.x, x1)),
        height=max(1, min(box.y + box.height, y2) - max(box.y, y1)),
    )
    return img[y1:y2, x1:x2].copy(), rel


def build_cache(paths: list[Path]) -> dict[str, np.ndarray]:
    """Lê as imagens e monta os arrays do cache (recortes achatados + formas, caixas, rótulos, tipos)."""
    crops, boxes, labels, tipos = [], [], [], []
    for caminho in paths:
        a = load_sample(caminho).annotation
        rotulo = encode(a.plate)
        img = cv2.imread(str(caminho))
        if rotulo is None or img is None:
            continue
        r = margin_crop(img, a.bbox)
        if r is None:
            continue
        crops.append(r[0])
        boxes.append([r[1].x, r[1].y, r[1].width, r[1].height])
        labels.append(rotulo)
        tipos.append(a.vehicle_type == "motorcycle")
    return {
        "flat": np.concatenate([c.ravel() for c in crops]),
        "shapes": np.array([c.shape for c in crops], np.int32),
        "boxes": np.array(boxes, np.int32),
        "labels": np.stack(labels),
        "moto": np.array(tipos, bool),
    }


class CropSet:
    """Recortes com margem em memória (do cache), com acesso por índice."""

    def __init__(self, d) -> None:
        self.flat, self.shapes, self.boxes = d["flat"], d["shapes"], d["boxes"]
        self.labels, self.moto = d["labels"], d["moto"]
        self.offsets = np.concatenate([[0], np.cumsum(np.prod(self.shapes, axis=1))])

    def __len__(self) -> int:
        return len(self.labels)

    def crop(self, i: int) -> np.ndarray:
        return self.flat[self.offsets[i] : self.offsets[i + 1]].reshape(self.shapes[i])

    def box(self, i: int) -> BoundingBox:
        x, y, w, h = (int(v) for v in self.boxes[i])
        return BoundingBox(x=x, y=y, width=w, height=h)

    def vehicle_type(self, i: int) -> str:
        return "motorcycle" if self.moto[i] else "car"


def cache_name(split: str, per_folder: int) -> str:
    """Chave do cache: inclui o split e o per_folder (0 = todas as imagens)."""
    return f"ocr_{split}_pf{per_folder}_m{int(MARGIN * 100)}"


def load_cached(name: str, paths_fn, cache_dir: Path = CACHE_DIR) -> CropSet:
    """Lê `cache_dir/name.npz`; se não existir, lê as imagens (`paths_fn()`) e cria o cache."""
    arquivo = cache_dir / f"{name}.npz"
    if arquivo.is_file():
        with np.load(arquivo) as d:
            return CropSet({k: d[k] for k in d.files})
    d = build_cache(paths_fn())
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = arquivo.with_name(name + ".tmp.npz")
    np.savez(tmp, **d)
    tmp.replace(arquivo)
    return CropSet(d)


def augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Augmentação de UMA imagem (48, 128, 1) em [0, 1]: brilho, contraste, desfoque e ruído leves."""
    media = x.mean()
    x = (x - media) * rng.uniform(0.7, 1.3) + media + rng.uniform(-0.15, 0.15)
    if rng.random() < 0.3:
        x = cv2.GaussianBlur(x[:, :, 0], (0, 0), rng.uniform(0.3, 1.2))[:, :, None]
    if rng.random() < 0.3:
        x = x + rng.normal(0, rng.uniform(0.005, 0.03), x.shape)
    return np.clip(x, 0, 1).astype(np.float32)


def train_image(cs: CropSet, i: int, rng: np.random.Generator) -> np.ndarray:
    """Um exemplo de treino: caixa tremida sorteada agora, recorte, tamanho do OCR e augmentação."""
    crop = cs.crop(i)
    recorte = crop_region(crop, jitter_box(cs.box(i), crop.shape, rng))
    if recorte is None or min(recorte.shape[:2]) < 2:
        recorte = crop_region(crop, cs.box(i))
    return augment(ocr_input(recorte, cs.vehicle_type(i))[0], rng)


def val_images(cs: CropSet) -> np.ndarray:
    """Validação: caixa do gabarito, sem tremor nem augmentação."""
    return np.stack(
        [ocr_input(crop_region(cs.crop(i), cs.box(i)), cs.vehicle_type(i))[0] for i in range(len(cs))]
    )


def make_sequence(cs: CropSet, seed: int, batch: int = BATCH):
    """keras PyDataset: monta cada lote sorteando tremor e augmentação, imagem por imagem, a cada época."""
    import keras

    class JitterSequence(keras.utils.PyDataset):
        def __init__(self):
            super().__init__()
            self.rng = np.random.default_rng(seed)
            self.order = self.rng.permutation(len(cs))

        def __len__(self):
            return -(-len(cs) // batch)

        def __getitem__(self, idx):
            ids = self.order[idx * batch : (idx + 1) * batch]
            x = np.stack([train_image(cs, int(i), self.rng) for i in ids])
            return x, cs.labels[ids]

        def on_epoch_end(self):
            self.order = self.rng.permutation(len(cs))

    return JitterSequence()


def checkpoint_callback(save, monitor: str = "val_loss"):
    """Callback keras: chama `save(epoch, logs)` sempre que `monitor` melhora (menor é melhor)."""
    import keras

    class BestCheckpoint(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best = np.inf

        def on_epoch_end(self, epoch, logs=None):
            valor = (logs or {}).get(monitor)
            if valor is not None and valor < self.best:
                self.best = float(valor)
                save(epoch, logs)

    return BestCheckpoint()


def normalizar(x):
    """R3: padroniza cada imagem (média 0, desvio 1) dentro do modelo; camada nativa do Keras."""
    from keras import layers

    return layers.LayerNormalization(axis=[1, 2, 3], center=False, scale=False, epsilon=1e-4)(x)


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
    x = normalizar(entrada)
    x = bloco(x, 32)  # 24x64
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
    x = conv(normalizar(entrada), 32)
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
    ap.add_argument("--val-per-folder", type=int, default=0, help="0 = todas as imagens de validação")
    ap.add_argument("--copies", type=int, default=1, help="obsoleto, ignorado (o tremor agora é sorteado a cada época)")
    ap.add_argument("--epochs", type=int, default=15, help="máximo; a parada antecipada pode acabar antes")
    ap.add_argument("--out", type=Path, default=OCR_PATH)
    args = ap.parse_args()

    import keras

    keras.utils.set_random_seed(SEED)
    divisao = split_paths(find_images(RAIZ))
    tr = load_cached(
        cache_name("train", args.per_folder), lambda: pick(divisao["train"], args.per_folder)
    )
    va = load_cached(
        cache_name("val", args.val_per_folder),
        lambda: pick(divisao["val"], args.val_per_folder or 10**9),
    )
    x_va, y_va = val_images(va), va.labels
    print("treino", len(tr), "validação", x_va.shape, flush=True)
    treino = make_sequence(tr, SEED)

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

    params = {
        "arch": args.arch,
        "per_folder": args.per_folder,
        "epochs_max": args.epochs,
        "seed": SEED,
        "cache_margin": MARGIN,
    }
    dataset = {"train": len(tr), "val": len(va)}

    def salvar(epoch, logs):
        """Checkpoint: modelo de inferência + ficha, sempre que val_loss melhora."""
        args.out.parent.mkdir(parents=True, exist_ok=True)
        inferencia.save(args.out)
        metricas = {k: float(v) for k, v in logs.items() if k.startswith(("val_", "loss"))}
        write_card(
            args.out,
            "ocr",
            params={**params, "epochs_rodadas": epoch + 1},
            metrics={"best_epoch": epoch + 1, **{f"best_{k}": v for k, v in metricas.items()}},
            dataset=dataset,
        )
        print(f"  checkpoint salvo (época {epoch + 1}, val_loss={logs['val_loss']:.4f})", flush=True)

    historico = treinavel.fit(
        treino,
        validation_data=(x_va, y_va),
        epochs=args.epochs,
        callbacks=[
            ValScores(),  # antes do checkpoint: a ficha leva as métricas de placa
            checkpoint_callback(salvar),
            keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ],
        verbose=2,
    )
    h = historico.history
    melhor = int(np.argmin(h["val_loss"]))
    # O arquivo já guarda o melhor modelo (checkpoint); aqui só se completa a ficha com o total de épocas.
    write_card(
        args.out,
        "ocr",
        params={**params, "epochs_rodadas": len(h["val_loss"])},
        metrics={
            "best_epoch": melhor + 1,
            **{f"best_{k}": float(v[melhor]) for k, v in h.items() if k.startswith(("val_", "loss"))},
        },
        dataset=dataset,
    )
    print("Salvo em", args.out, "(ficha:", args.out.with_suffix(".json"), ") melhor época", melhor + 1)


if __name__ == "__main__":
    main()
