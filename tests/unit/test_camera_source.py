import cv2
import numpy as np
import pytest

from app.camera.source import CameraOpenError, CameraReadError, CameraSource


class FakeCapture:
    """Câmera falsa: entrega uma lista de frames e conta quantas vezes release() foi chamado."""

    def __init__(self, n_frames=10, fps=5.0, abre=True):
        self._abre = abre
        self._fps = fps
        self._frames = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(n_frames)]
        self.release_calls = 0

    def isOpened(self):
        return self._abre

    def read(self):
        if self._frames:
            return True, self._frames.pop(0)
        return False, None

    def get(self, prop):
        return self._fps if prop == cv2.CAP_PROP_FPS else 0.0

    def release(self):
        self.release_calls += 1


def criar(fake, fonte="data/teste.mp4", intervalo=1.0):
    return CameraSource(fonte, intervalo, capture_factory=lambda _fonte: fake)


def test_falha_ao_abrir_levanta_erro_e_libera():
    fake = FakeCapture(abre=False)
    with pytest.raises(CameraOpenError):
        list(criar(fake).frames())
    assert fake.release_calls == 1


def test_arquivo_termina_sem_erro_e_amostra_os_frames_certos():
    fake = FakeCapture(n_frames=10, fps=5.0)  # intervalo 1 s -> passo 5 -> frames 0 e 5
    emitidos = list(criar(fake).frames())
    assert [int(f[0, 0, 0]) for f in emitidos] == [0, 5]
    assert fake.release_calls == 1


def test_fonte_ao_vivo_levanta_erro_quando_para_de_entregar():
    fake = FakeCapture(n_frames=3, fps=5.0)
    with pytest.raises(CameraReadError):
        list(criar(fake, fonte="0").frames())
    assert fake.release_calls == 1


def test_libera_quando_o_consumidor_fecha_o_gerador_antes_do_fim():
    fake = FakeCapture(n_frames=10, fps=5.0)
    gerador = criar(fake).frames()
    next(gerador)
    gerador.close()
    assert fake.release_calls == 1


def test_stop_encerra_o_gerador_e_libera():
    fake = FakeCapture(n_frames=10, fps=5.0)
    camera = criar(fake)
    gerador = camera.frames()
    next(gerador)
    camera.stop()
    assert list(gerador) == []
    assert fake.release_calls == 1


def test_close_duas_vezes_e_seguro():
    fake = FakeCapture()
    camera = criar(fake)
    camera.open()
    camera.close()
    camera.close()
    assert fake.release_calls == 1