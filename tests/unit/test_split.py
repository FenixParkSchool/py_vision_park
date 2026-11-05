from pathlib import Path

import pytest

from training.split import split_of, split_paths


def test_blocos_60_20_20():
    nomes = [split_of(i, 10) for i in range(10)]
    assert nomes == ["train"] * 6 + ["val"] * 2 + ["test"] * 2


def test_indice_fora_levanta_erro():
    with pytest.raises(ValueError):
        split_of(10, 10)


def test_divide_cada_pasta_separadamente_e_sem_sobrepor():
    paths = [Path(f"r/{pasta}/img_{i:03d}.jpg") for pasta in ("a", "b") for i in range(10)]
    divisao = split_paths(list(reversed(paths)))  # ordem de entrada não importa
    assert len(divisao["train"]) == 12 and len(divisao["val"]) == 4 and len(divisao["test"]) == 4
    assert divisao["test"][0] == Path("r/a/img_008.jpg")
    todos = divisao["train"] + divisao["val"] + divisao["test"]
    assert sorted(todos) == sorted(paths)
