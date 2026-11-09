import logging
import time
from collections.abc import Callable

import httpx

from app.schemas.events import PlateEvent

logger = logging.getLogger(__name__)

ENDPOINT = "/eventos-alpr"
# 409 = corrida com o mesmo eventId (reenviar devolve 200); 429 = rate limit; 5xx = falha do servidor.
# 503 fica de fora: no Back_park significa "ALPR_API_KEY não configurada", e repetir não resolve.
_RETRY_STATUS = {409, 429}


MAX_RETRY_AFTER_S = 30.0  # teto: nunca travar o worker por causa de um header exagerado


def _retry_after(resp: httpx.Response) -> float | None:
    """Lê Retry-After em segundos (o formato data HTTP é ignorado). None se ausente/inválido."""
    try:
        valor = float(resp.headers["Retry-After"])
    except (KeyError, ValueError):
        return None
    return min(max(valor, 0.0), MAX_RETRY_AFTER_S)


class BackendError(Exception):
    pass


class BackendRejected(BackendError):
    """O backend recusou o evento (400, 401, 503...). Repetir não adianta."""


class BackendUnavailable(BackendError):
    """Esgotaram as tentativas (rede, timeout, 429, 5xx)."""


def _retryable(status: int) -> bool:
    return status in _RETRY_STATUS or (status >= 500 and status != 503)


class BackendClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 5,
        max_retries: int = 3,
        backoff_s: float = 0.5,
        transport: httpx.BaseTransport | None = None,  # injetado nos testes
        sleep: Callable[[float], None] = time.sleep,  # injetado nos testes
    ) -> None:
        if not api_key.strip():
            raise ValueError("backend_api_key vazia: configure BACKEND_API_KEY")
        self._max_retries = max_retries
        self._backoff_s = backoff_s
        self._sleep = sleep
        # Um único Client reaproveita a conexão entre envios.
        self._client = httpx.Client(
            base_url=base_url,
            headers={"X-API-Key": api_key},  # nunca logar este header
            timeout=timeout,
            transport=transport,
        )

    def send(self, event: PlateEvent) -> bool:
        """Envia o evento. Devolve True se o backend já o conhecia (duplicado)."""
        payload = event.to_payload()  # o mesmo eventId vai em todas as tentativas
        reason = "nenhuma tentativa"
        retry_after: float | None = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = self._client.post(ENDPOINT, json=payload)
            except httpx.TransportError as exc:  # timeout, conexão recusada, DNS...
                reason = type(exc).__name__
                retry_after = None
            else:
                if resp.status_code in (200, 201):
                    return bool(resp.json().get("duplicado", False))
                if not _retryable(resp.status_code):
                    raise BackendRejected(f"HTTP {resp.status_code}: {resp.text[:200]}")
                reason = f"HTTP {resp.status_code}"
                retry_after = _retry_after(resp) if resp.status_code == 429 else None

            if attempt < self._max_retries:
                logger.warning("envio falhou (%s); tentativa %d", reason, attempt + 1)
                espera = self._backoff_s * 2**attempt  # 0.5s, 1s, 2s...
                self._sleep(max(espera, retry_after) if retry_after is not None else espera)
        raise BackendUnavailable(f"sem sucesso após {self._max_retries + 1} tentativas: {reason}")

    def close(self) -> None:
        self._client.close()
