"""Normalização e validação do formato de placa brasileira.

Formatos aceitos (7 caracteres, sem hífen):
- Mercosul: ABC1D23 (3 letras, 1 dígito, 1 letra, 2 dígitos)
- Antiga:   ABC1234 (3 letras, 4 dígitos)

Não troca caracteres parecidos (O/0, I/1, B/8...): se o OCR leu "ABCID23",
o resultado é inválido, e não "ABC1D23". O texto cru é sempre preservado.
"""

import re
from dataclasses import dataclass
from typing import Literal

PlateFormat = Literal["mercosul", "antiga"]

_MERCOSUL = re.compile(r"[A-Z]{3}[0-9][A-Z][0-9]{2}")
_ANTIGA = re.compile(r"[A-Z]{3}[0-9]{4}")
# Só separadores visuais são removidos; qualquer outro caractere fica e invalida a placa.
_SEPARADORES = re.compile(r"[\s\-.·]")


@dataclass(frozen=True)
class NormalizedPlate:
    raw: str  # exatamente o que o OCR devolveu
    text: str  # maiúsculo, sem espaços/hífen/ponto
    format: PlateFormat | None  # None = não bate com nenhum formato

    @property
    def is_valid(self) -> bool:
        return self.format is not None


def normalize_plate(raw: str) -> NormalizedPlate:
    texto = _SEPARADORES.sub("", raw).upper()
    if _MERCOSUL.fullmatch(texto):
        formato: PlateFormat | None = "mercosul"
    elif _ANTIGA.fullmatch(texto):
        formato = "antiga"
    else:
        formato = None
    return NormalizedPlate(raw=raw, text=texto, format=formato)
