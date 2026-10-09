import cv2
import numpy as np
import pytest

from app.camera.source import CameraOpenError, CameraSource


def gerar_video(caminho, n_frames=30, fps=10, largura=64, altura=48):
    # VideoWriter recebe o tamanho como (largura, altura), ordem inversa do .shape
    escritor = cv2.VideoWriter(
        str(caminho), cv2.VideoWriter_fourcc(*"MJPG"), fps, (largura, altura)
    )
    assert escritor.isOpened()
    for i in range(n_frames):
        escritor.write(np.full((altura, largura, 3), (i * 8) % 256, dtype=np.uint8))
    escritor.release()


def test_le_e_amostra_um_video_de_verdade(tmp_path):
    video = tmp_path / "teste.avi"
    gerar_video(video)  # 30 frames a 10 fps; intervalo 1 s -> passo 10 -> frames 0, 10 e 20
    frames = list(CameraSource(str(video), sample_interval_s=1.0).frames())
    assert len(frames) == 3
    assert frames[0].shape == (48, 64, 3)


def test_arquivo_inexistente_levanta_erro_claro(tmp_path):
    camera = CameraSource(str(tmp_path / "nao_existe.avi"), sample_interval_s=1.0)
    with pytest.raises(CameraOpenError):
        list(camera.frames())