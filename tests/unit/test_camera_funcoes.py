import pytest

from app.camera.source import compute_stride, describe_source, parse_source


def test_parse_source_indice_de_webcam():
    assert parse_source("0") == 0
    assert parse_source(" 2 ") == 2


def test_parse_source_caminho_e_url_continuam_texto():
    assert parse_source("data/teste.mp4") == "data/teste.mp4"
    assert parse_source("rtsp://u:p@10.0.0.5:554/s") == "rtsp://u:p@10.0.0.5:554/s"
    assert parse_source("12abc") == "12abc"


def test_parse_source_vazio_levanta_erro():
    with pytest.raises(ValueError):
        parse_source("   ")


def test_describe_source_esconde_credenciais():
    texto = describe_source("rtsp://admin:segredo@192.168.0.50:554/stream1")
    assert "segredo" not in texto
    assert "admin" not in texto
    assert "192.168.0.50" in texto


def test_describe_source_webcam_e_arquivo():
    assert describe_source(0) == "webcam #0"
    assert describe_source("data/teste.mp4") == "data/teste.mp4"


@pytest.mark.parametrize(
    "fps, intervalo, esperado",
    [
        (30, 1.0, 30),
        (30, 0.5, 15),
        (30, 0.001, 1),
        (0, 1.0, 30),
        (float("nan"), 1.0, 30),
        (-5, 2.0, 60),
    ],
)
def test_compute_stride(fps, intervalo, esperado):
    assert compute_stride(fps, intervalo) == esperado