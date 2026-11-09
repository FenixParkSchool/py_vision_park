"""Ficha de versão de um modelo: um JSON ao lado do arquivo .keras.

O treino grava (`write_card`); o serviço só lê (`describe`) para registrar no log qual
modelo está rodando. Não importa TensorFlow: as versões vêm de importlib.metadata.
"""

import hashlib
import json
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path


def card_path(model_path: Path) -> Path:
    return Path(model_path).with_suffix(".json")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def _version(pacote: str) -> str | None:
    try:
        return metadata.version(pacote)
    except metadata.PackageNotFoundError:
        return None


def write_card(model_path: Path, kind: str, params: dict, metrics: dict, dataset: dict) -> Path:
    """Grava `<modelo>.json` com versão, parâmetros, métricas e tamanho dos conjuntos."""
    ficha = {
        "kind": kind,  # "detector" ou "ocr"
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sha256": file_sha256(model_path),
        "tensorflow": _version("tensorflow"),
        "keras": _version("keras"),
        "params": params,
        "metrics": metrics,
        "dataset": dataset,
    }
    destino = card_path(model_path)
    destino.write_text(json.dumps(ficha, indent=2, ensure_ascii=False), encoding="utf-8")
    return destino


def read_card(model_path: Path) -> dict | None:
    arquivo = card_path(model_path)
    if not arquivo.is_file():
        return None
    try:
        return json.loads(arquivo.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def describe(model_path: Path) -> str:
    """Uma linha para log: versão do modelo, sem caminhos sensíveis."""
    ficha = read_card(model_path)
    if ficha is None:
        return f"{Path(model_path).name} (sem ficha de versão)"
    return f"{Path(model_path).name} sha256={ficha['sha256'][:12]} criado={ficha['created_at']}"
