import json
import time

import httpx
import numpy as np

from app.camera.source import CameraReadError
from app.recognition.base import PlateDetection, PlateReading
from app.recognition.fakes import FakePlateDetector, FakePlateReader
from app.schemas.recognition import BoundingBox
from app.services.backend_client import BackendClient
from app.services.deduplicator import Deduplicator
from app.services.pipeline import RecognitionPipeline
from app.services.worker import AlprWorker


class FakeCamera:
    def __init__(self, n_frames=3, error=None, block=False):
        self.n_frames, self.error, self.block = n_frames, error, block
        self.opened = self.closed = self.stopped = False

    def open(self):
        self.opened = True

    def frames(self):
        for _ in range(self.n_frames):
            yield np.zeros((20, 20, 3), dtype=np.uint8)
        if self.error:
            raise self.error
        while self.block and not self.stopped:
            time.sleep(0.005)

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


def make_pipeline(detector=None):
    det = detector or FakePlateDetector(
        [PlateDetection(BoundingBox(x=2, y=2, width=10, height=10), 0.9)]
    )
    return RecognitionPipeline(
        det, FakePlateReader(PlateReading("ABC1D23", 0.9)), "cam-1", 0.5, Deduplicator(60)
    )


def make_client(handler):
    return BackendClient("http://b", "key", max_retries=0, transport=httpx.MockTransport(handler),
                         sleep=lambda s: None)


def wait_done(worker):
    worker._thread.join(timeout=3)
    assert not worker.running


def test_envio_feliz():
    bodies = []

    def handler(req):
        bodies.append(json.loads(req.content))
        return httpx.Response(201, json={"duplicado": False})

    w = AlprWorker(FakeCamera(3), make_pipeline(), make_client(handler))
    w.start()
    wait_done(w)
    assert w.frames == 3
    assert w.events_sent == 1  # dedup do pipeline suprime as repetições
    assert bodies[0]["plateText"] == "ABC1D23"
    assert w.last_error is None


def test_duplicado_do_backend_contado():
    w = AlprWorker(FakeCamera(1), make_pipeline(),
                   make_client(lambda r: httpx.Response(200, json={"duplicado": True})))
    w.start()
    wait_done(w)
    assert w.duplicates == 1 and w.events_sent == 0


def test_falha_de_envio_nao_derruba():
    w = AlprWorker(FakeCamera(1, block=True), make_pipeline(),
                   make_client(lambda r: httpx.Response(500)))
    w.start()
    time.sleep(0.3)
    assert w.send_failures == 1
    assert w.running and w.ready and w.last_error is None
    w.stop()
    assert not w.running


def test_erro_de_camera_deixa_ready_falso():
    cam = FakeCamera(1, error=CameraReadError("cabo USB removido"))
    w = AlprWorker(cam, make_pipeline(), make_client(lambda r: httpx.Response(201, json={})))
    w.start()
    wait_done(w)
    assert not w.ready
    assert "CameraReadError" in w.last_error


def test_erro_do_detector_para_o_worker():
    class Boom:
        def detect(self, frame):
            raise RuntimeError("falha do modelo")

    w = AlprWorker(FakeCamera(1), make_pipeline(Boom()),
                   make_client(lambda r: httpx.Response(201, json={})))
    w.start()
    wait_done(w)
    assert not w.ready and "RuntimeError" in w.last_error


def test_stop_encerra_thread_e_fecha_camera():
    cam = FakeCamera(0, block=True)
    w = AlprWorker(cam, make_pipeline(), make_client(lambda r: httpx.Response(201, json={})))
    w.start()
    time.sleep(0.1)
    assert w.ready
    w.stop()
    assert cam.stopped and cam.closed and not w.running and not w.ready


def test_placa_de_formato_invalido_nao_e_enviada():
    chamadas = []
    pipeline = RecognitionPipeline(
        FakePlateDetector([PlateDetection(BoundingBox(x=2, y=2, width=10, height=10), 0.9)]),
        FakePlateReader(PlateReading("ABCID23", 0.9)),  # I no lugar do 1: não é corrigido
        "cam-1", 0.5, Deduplicator(60),
    )
    w = AlprWorker(FakeCamera(1), pipeline,
                   make_client(lambda r: chamadas.append(r) or httpx.Response(201, json={})))
    w.start()
    wait_done(w)
    assert chamadas == []
    assert w.invalid_plates == 1 and w.events_sent == 0
