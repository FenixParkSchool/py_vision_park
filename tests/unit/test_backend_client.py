from datetime import UTC, datetime

import httpx
import pytest

from app.schemas.events import PlateEvent
from app.schemas.recognition import PlateObservation
from app.services.backend_client import BackendClient, BackendRejected, BackendUnavailable

KEY = "k" * 32


def _event(plate="abc-1d23"):
    obs = PlateObservation(
        camera_id="portaria-1",
        plate_text=plate,
        detection_confidence=0.9,
        ocr_confidence=0.8,
        captured_at=datetime(2026, 11, 6, 12, 0, tzinfo=UTC),
    )
    return PlateEvent.from_observation(obs, event_id="evento-0001")


def _client(handler, **kw):
    sleeps = []
    c = BackendClient(
        "http://backend", KEY, transport=httpx.MockTransport(handler),
        sleep=sleeps.append, **kw,
    )
    return c, sleeps


def test_payload_camelcase_sem_bbox_e_placa_normalizada():
    p = _event().to_payload()
    assert p == {
        "eventId": "evento-0001",
        "cameraId": "portaria-1",
        "plateText": "ABC1D23",
        "detectionConfidence": 0.9,
        "ocrConfidence": 0.8,
        "capturedAt": "2026-11-06T12:00:00Z",
    }


def test_envia_com_header_e_rota():
    vistos = []

    def handler(req):
        vistos.append(req)
        return httpx.Response(201, json={"duplicado": False})

    c, _ = _client(handler)
    assert c.send(_event()) is False
    assert vistos[0].url.path == "/eventos-alpr"
    assert vistos[0].headers["X-API-Key"] == KEY


def test_200_duplicado():
    c, _ = _client(lambda r: httpx.Response(200, json={"duplicado": True}))
    assert c.send(_event()) is True


def test_retry_em_5xx_reusa_event_id_e_depois_sucesso():
    respostas = [httpx.Response(500), httpx.Response(429), httpx.Response(201, json={})]
    corpos = []

    def handler(req):
        corpos.append(req.content)
        return respostas.pop(0)

    c, sleeps = _client(handler)
    c.send(_event())
    assert len(corpos) == 3 and len(set(corpos)) == 1
    assert sleeps == [0.5, 1.0]


def test_retry_em_erro_de_rede():
    estado = {"n": 0}

    def handler(req):
        estado["n"] += 1
        if estado["n"] == 1:
            raise httpx.ConnectError("recusada")
        return httpx.Response(201, json={})

    c, _ = _client(handler)
    c.send(_event())
    assert estado["n"] == 2


@pytest.mark.parametrize("status", [400, 401, 503])
def test_nao_repete_em_erro_definitivo(status):
    chamadas = []

    def handler(req):
        chamadas.append(1)
        return httpx.Response(status, json={"error": "x"})

    c, _ = _client(handler)
    with pytest.raises(BackendRejected):
        c.send(_event())
    assert len(chamadas) == 1


def test_esgota_tentativas():
    c, sleeps = _client(lambda r: httpx.Response(500), max_retries=2)
    with pytest.raises(BackendUnavailable):
        c.send(_event())
    assert len(sleeps) == 2


def test_recusa_chave_vazia():
    with pytest.raises(ValueError):
        BackendClient("http://backend", "  ")
