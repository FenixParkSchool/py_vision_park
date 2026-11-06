import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response

from app.camera.source import CameraSource
from app.config import get_settings
from app.recognition.base import PlateDetector, PlateReader
from app.recognition.fakes import FakePlateDetector, FakePlateReader
from app.services.backend_client import BackendClient
from app.services.deduplicator import Deduplicator
from app.services.pipeline import RecognitionPipeline
from app.services.worker import AlprWorker

log = logging.getLogger(__name__)


def build_pipeline(settings, deduplicator: Deduplicator) -> RecognitionPipeline:
    # TODO: trocar os fakes pelo detector/OCR reais quando o passo 1 estiver integrado.
    detector: PlateDetector = FakePlateDetector()
    reader: PlateReader = FakePlateReader(None)
    return RecognitionPipeline(
        detector, reader, settings.camera_id, settings.min_confidence, deduplicator
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.worker = None
    app.state.startup_error = None
    client = None
    if not settings.backend_api_key.strip():
        app.state.startup_error = "BACKEND_API_KEY não configurada"
        log.error(app.state.startup_error)
    else:
        camera = CameraSource(settings.camera_source, settings.sample_interval_seconds)
        client = BackendClient(
            settings.backend_url,
            settings.backend_api_key,
            timeout=settings.backend_timeout,
            max_retries=settings.backend_max_retries,
        )
        pipeline = build_pipeline(settings, Deduplicator(settings.dedup_seconds))
        app.state.worker = AlprWorker(camera, pipeline, client)
        app.state.worker.start()
    try:
        yield
    finally:
        if app.state.worker is not None:
            app.state.worker.stop()
        if client is not None:
            client.close()


app = FastAPI(
    title="Microserviço ALPR - Estacionamento",
    description="Leitura contínua de placas por câmera USB, enviando eventos ao backend.",
    lifespan=lifespan,
)


@app.get("/health/live")
def live() -> dict:
    return {"status": "live"}


@app.get("/health/ready")
def ready(response: Response) -> dict:
    worker = app.state.worker
    if worker is None:
        response.status_code = 503
        return {"status": "not_ready", "reason": app.state.startup_error or "worker não iniciado"}
    if not worker.ready:
        response.status_code = 503
        return {"status": "not_ready", "reason": worker.last_error or "câmera não aberta ou worker parado"}
    return {
        "status": "ready",
        "frames": worker.frames,
        "events_sent": worker.events_sent,
        "duplicates": worker.duplicates,
        "send_failures": worker.send_failures,
    }
