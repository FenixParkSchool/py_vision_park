import time
from collections.abc import Callable


class Deduplicator:
    """Evita reportar a mesma placa, na mesma câmera, várias vezes seguidas.

    Janela deslizante: cada nova visão da placa renova o prazo. Uma placa só volta a
    ser "nova" depois de `window_s` segundos sem ser vista. O estado fica só na
    memória: é perdido ao reiniciar o serviço e não é compartilhado entre processos.
    
    """

    def __init__(self, window_s: float, clock: Callable[[], float] = time.monotonic) -> None:
        if window_s <= 0:
            raise ValueError("A janela de deduplicação deve ser maior que zero")
        self._window_s = window_s
        self._clock = clock
        self._last_seen: dict[tuple[str, str], float] = {}

    def __len__(self) -> int:
        return len(self._last_seen)

    def is_duplicate(self, camera_id: str, plate: str) -> bool:
        """Diz se a placa já foi vista na janela. Também registra esta visão."""
        agora = self._clock()
        self._forget_expired(agora)
        chave = (camera_id, plate)
        duplicada = chave in self._last_seen
        self._last_seen[chave] = agora
        return duplicada

    def _forget_expired(self, agora: float) -> None:
        expiradas = [c for c, visto in self._last_seen.items() if agora - visto >= self._window_s]
        for chave in expiradas:
            del self._last_seen[chave]