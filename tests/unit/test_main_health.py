import time

from fastapi.testclient import TestClient

import app.main as main
from app.camera.source import CameraOpenError
from app.config import Settings


class FakeCamera:
    def __init__(self, *a, **k):
        self.stopped = False

    def open(self):
        pass

    def frames(self):
        while not self.stopped:
            time.sleep(0.005)
        return
        yield

    def stop(self):
        self.stopped = True

    def close(self):
        pass


class CameraQueNaoAbre(FakeCamera):
    def open(self):
        raise CameraOpenError("sem câmera")


def _settings(key):
    return lambda: Settings(backend_api_key=key, _env_file=None)


def test_live_200_e_ready_503_sem_api_key(monkeypatch):
    monkeypatch.setattr(main, "get_settings", _settings(""))
    with TestClient(main.app) as client:
        assert client.get("/health/live").status_code == 200
        r = client.get("/health/ready")
        assert r.status_code == 503
        assert "BACKEND_API_KEY" in r.json()["reason"]


def test_ready_503_quando_worker_nao_esta_pronto(monkeypatch):
    monkeypatch.setattr(main, "get_settings", _settings("k"))
    monkeypatch.setattr(main, "CameraSource", CameraQueNaoAbre)
    with TestClient(main.app) as client:
        time.sleep(0.2)
        assert client.get("/health/live").status_code == 200
        r = client.get("/health/ready")
        assert r.status_code == 503
        assert "CameraOpenError" in r.json()["reason"]


def test_ready_200_com_camera_aberta(monkeypatch):
    monkeypatch.setattr(main, "get_settings", _settings("k"))
    monkeypatch.setattr(main, "CameraSource", FakeCamera)
    with TestClient(main.app) as client:
        time.sleep(0.2)
        r = client.get("/health/ready")
        assert r.status_code == 200
        assert r.json()["invalid_plates"] == 0


def test_modo_tf_sem_modelo_deixa_ready_503(monkeypatch, tmp_path):
    def settings():
        return Settings(
            backend_api_key="k" * 32,
            recognition_mode="tf",
            detector_model_path=str(tmp_path / "nao_existe.keras"),
            ocr_model_path=str(tmp_path / "nao_existe2.keras"),
        )

    monkeypatch.setattr(main, "get_settings", settings)
    monkeypatch.setattr(main, "CameraSource", FakeCamera)
    with TestClient(main.app) as client:
        r = client.get("/health/ready")
        assert r.status_code == 503
        assert "Modelo não encontrado" in r.json()["reason"]
