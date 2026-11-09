"""Teste ponta a ponta contra um Back_park de verdade. Só roda se as variáveis existirem:

    E2E_BACKEND_URL=http://localhost:3099  E2E_BACKEND_API_KEY=<ALPR_API_KEY do backend>
    .venv\\Scripts\\python -m pytest tests/integration/test_backend_real.py -q

Grava eventos reais no banco do backend (eventId começa com "teste-e2e-"); o backend não tem rota de
exclusão, então apague depois (SQL: DELETE FROM evento_alpr WHERE event_id LIKE 'teste-e2e-%').
"""

import os
import time
from datetime import UTC, datetime

import pytest

from app.schemas.events import PlateEvent
from app.schemas.recognition import PlateObservation
from app.services.backend_client import BackendClient, BackendRejected

URL = os.getenv("E2E_BACKEND_URL")
KEY = os.getenv("E2E_BACKEND_API_KEY")

pytestmark = pytest.mark.skipif(not (URL and KEY), reason="defina E2E_BACKEND_URL e E2E_BACKEND_API_KEY")


def _evento(sufixo: str) -> PlateEvent:
    obs = PlateObservation(
        camera_id="cam-e2e",
        plate_text="abc-1d23",
        detection_confidence=0.91,
        ocr_confidence=0.87,
        captured_at=datetime.now(UTC),
    )
    return PlateEvent.from_observation(obs, event_id=f"teste-e2e-{int(time.time())}-{sufixo}")


def test_envio_real_e_idempotencia():
    client = BackendClient(URL, KEY)
    try:
        evento = _evento("ok")
        assert client.send(evento) is False  # 201: criado
        assert client.send(evento) is True  # 200: o backend já conhecia (duplicado)
    finally:
        client.close()


def test_chave_errada_e_rejeitada_sem_retry():
    client = BackendClient(URL, "x" * 32, max_retries=3, sleep=lambda s: pytest.fail("não deveria repetir"))
    try:
        with pytest.raises(BackendRejected):
            client.send(_evento("chave"))
    finally:
        client.close()
