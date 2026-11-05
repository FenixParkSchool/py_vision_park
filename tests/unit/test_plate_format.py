import pytest

from app.recognition.plate_format import normalize_plate


@pytest.mark.parametrize(
    ("raw", "texto", "formato"),
    [
        ("ABC1D23", "ABC1D23", "mercosul"),
        ("ABC1234", "ABC1234", "antiga"),
        ("abc-1234", "ABC1234", "antiga"),
        (" abc 1d23 ", "ABC1D23", "mercosul"),
        ("ABC.1234", "ABC1234", "antiga"),
    ],
)
def test_formatos_validos(raw, texto, formato):
    placa = normalize_plate(raw)
    assert placa.text == texto
    assert placa.format == formato
    assert placa.is_valid
    assert placa.raw == raw  # texto cru preservado


@pytest.mark.parametrize(
    "raw",
    [
        "ABCID23",  # I no lugar de 1: não é corrigido à força
        "AB0123",  # curto
        "ABC12345",  # longo
        "0BC1234",  # dígito onde deveria haver letra
        "ABC1D2O",  # letra O no lugar de 0
        "ABC_1234",  # caractere estranho não é removido
        "",
    ],
)
def test_invalidos_nao_sao_corrigidos(raw):
    placa = normalize_plate(raw)
    assert placa.format is None
    assert not placa.is_valid
    assert placa.raw == raw


def test_ambiguo_continua_como_lido():
    assert normalize_plate("abcid23").text == "ABCID23"
