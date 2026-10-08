import pytest
from pydantic import ValidationError

from app.config import Settings

# monkeypatch é uma ferramenta do pytest que muda variáveis de ambiente só durante aquele teste, e desfaz depois.


def test_valores_padrao(monkeypatch):
    for nome in ("CAMERA_SOURCE", "SAMPLE_INTERVAL_SECONDS", "MIN_CONFIDENCE"):
        monkeypatch.delenv(nome, raising=False)
    s = Settings(_env_file=None)
    assert s.camera_source == "data/teste.mp4"
    assert s.sample_interval_seconds == 1.0


def test_variavel_de_ambiente_sobrescreve_o_padrao(monkeypatch):
    monkeypatch.setenv("CAMERA_SOURCE", "0")
    assert Settings(_env_file=None).camera_source == "0"


def test_intervalo_zero_e_rejeitado(monkeypatch):
    monkeypatch.setenv("SAMPLE_INTERVAL_SECONDS", "0")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_confianca_acima_de_1_e_rejeitada(monkeypatch):
    monkeypatch.setenv("MIN_CONFIDENCE", "1.5")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)