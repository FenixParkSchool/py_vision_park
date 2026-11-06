import re
import uuid

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from app.schemas.recognition import PlateObservation


class PlateEvent(BaseModel):
    """Evento enviado ao backend (POST /eventos-alpr). Os campos seguem o schema estrito do Back_park."""

    # alias_generator: no JSON os nomes saem em camelCase (eventId, plateText...)
    # populate_by_name: no Python continuamos criando com snake_case
    model_config = ConfigDict(frozen=True, alias_generator=to_camel, populate_by_name=True)

    event_id: str = Field(min_length=8, max_length=64)
    camera_id: str = Field(min_length=1, max_length=64)
    plate_text: str = Field(min_length=1, max_length=20)
    detection_confidence: float = Field(ge=0, le=1)
    ocr_confidence: float = Field(ge=0, le=1)
    captured_at: AwareDatetime

    @classmethod
    def from_observation(cls, obs: PlateObservation, event_id: str | None = None) -> "PlateEvent":
        # O backend só aceita A-Z e 0-9 (e normaliza). Fazemos o mesmo aqui só para não tomar 400.
        # A validação real de formato Mercosul/antigo continua sendo o passo 1.
        plate = re.sub(r"[^A-Z0-9]", "", obs.plate_text.upper())
        return cls(
            event_id=event_id or uuid.uuid4().hex,  # 32 caracteres, dentro de 8..64
            camera_id=obs.camera_id,
            plate_text=plate,  # vazio -> ValidationError (min_length=1)
            detection_confidence=obs.detection_confidence,
            ocr_confidence=obs.ocr_confidence,
            captured_at=obs.captured_at,
        )

    def to_payload(self) -> dict:
        # by_alias -> camelCase; mode="json" -> datetime vira string ISO 8601 (UTC sai com "Z")
        # O bbox não existe aqui de propósito: o backend rejeita campo extra com 400.
        return self.model_dump(by_alias=True, mode="json")
