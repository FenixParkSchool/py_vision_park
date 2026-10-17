import pytest

from app.services.deduplicator import Deduplicator


class RelogioFalso:
    def __init__(self):
        self.agora = 0.0

    def __call__(self):
        return self.agora


def criar(janela=30):
    relogio = RelogioFalso()
    return Deduplicator(janela, clock=relogio), relogio


def test_primeira_vez_nao_e_duplicada():
    dedup, _ = criar()
    assert dedup.is_duplicate("portaria-1", "ABC1D23") is False


def test_repeticao_dentro_da_janela_e_duplicada():
    dedup, relogio = criar(30)
    dedup.is_duplicate("portaria-1", "ABC1D23")
    relogio.agora = 29.9
    assert dedup.is_duplicate("portaria-1", "ABC1D23") is True


def test_exatamente_no_fim_da_janela_ja_e_nova():
    dedup, relogio = criar(30)
    dedup.is_duplicate("portaria-1", "ABC1D23")
    relogio.agora = 30.0
    assert dedup.is_duplicate("portaria-1", "ABC1D23") is False


def test_depois_da_janela_e_nova():
    dedup, relogio = criar(30)
    dedup.is_duplicate("portaria-1", "ABC1D23")
    relogio.agora = 31
    assert dedup.is_duplicate("portaria-1", "ABC1D23") is False


def test_janela_desliza_a_cada_nova_visao():
    dedup, relogio = criar(30)
    assert dedup.is_duplicate("portaria-1", "ABC1D23") is False  # t=0
    relogio.agora = 20
    assert dedup.is_duplicate("portaria-1", "ABC1D23") is True
    relogio.agora = 40  # 20 s depois da última visão (t=20)
    assert dedup.is_duplicate("portaria-1", "ABC1D23") is True
    relogio.agora = 70  # 30 s depois da última visão (t=40)
    assert dedup.is_duplicate("portaria-1", "ABC1D23") is False


def test_cameras_diferentes_nao_se_misturam():
    dedup, _ = criar()
    dedup.is_duplicate("portaria-1", "ABC1D23")
    assert dedup.is_duplicate("portaria-2", "ABC1D23") is False


def test_placas_diferentes_nao_se_misturam():
    dedup, _ = criar()
    dedup.is_duplicate("portaria-1", "ABC1D23")
    assert dedup.is_duplicate("portaria-1", "XYZ9K88") is False


def test_memoria_e_limpa_quando_as_placas_expiram():
    dedup, relogio = criar(30)
    for i in range(100):
        dedup.is_duplicate("portaria-1", f"AAA{i:04d}")
    assert len(dedup) == 100
    relogio.agora = 31
    dedup.is_duplicate("portaria-1", "ZZZ9Z99")
    assert len(dedup) == 1


@pytest.mark.parametrize("janela", [0, -5])
def test_janela_invalida_e_rejeitada(janela):
    with pytest.raises(ValueError):
        Deduplicator(janela)