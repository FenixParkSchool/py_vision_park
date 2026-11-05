"""Divisão treino/validação/teste do RodoSol-ALPR, determinística.

O pacote local não traz o split.txt oficial. Dividimos cada pasta em blocos
contíguos pela ordem dos arquivos (60% treino, 20% validação, 20% teste):
imagens vizinhas costumam ser parecidas, então blocos evitam que quase-cópias
caiam em treino e teste ao mesmo tempo.
"""

from collections import defaultdict
from pathlib import Path
from typing import Literal

Split = Literal["train", "val", "test"]
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2


def split_of(index: int, total: int) -> Split:
    if not 0 <= index < total:
        raise ValueError(f"Índice {index} fora de 0..{total - 1}")
    if index < int(total * TRAIN_FRAC):
        return "train"
    if index < int(total * (TRAIN_FRAC + VAL_FRAC)):
        return "val"
    return "test"


def split_paths(paths: list[Path]) -> dict[Split, list[Path]]:
    """Agrupa por pasta-mãe e divide cada pasta em blocos; listas em ordem estável."""
    por_pasta: dict[Path, list[Path]] = defaultdict(list)
    for p in sorted(paths):
        por_pasta[p.parent].append(p)
    saida: dict[Split, list[Path]] = {"train": [], "val": [], "test": []}
    for _, lista in sorted(por_pasta.items()):
        for i, p in enumerate(lista):
            saida[split_of(i, len(lista))].append(p)
    return saida
