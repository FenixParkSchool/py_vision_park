from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class BoundingBox(BaseModel):
    """Retângulo em pixels absolutos do frame; (x, y) é o canto superior esquerdo."""

    # deixa o objeto imutável, para evitar que o mesmo seja alterado depois de criado (ex: para não alterar a leitura de uma placa)
    model_config = ConfigDict(frozen=True)


    # Field(ge=0), gt=0, min_length=1: as regras de cada campo (maior ou igual a 0, maior que 0, pelo menos 1 caractere). Largura de caixa zero ou placa vazia são rejeitadas.
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class PlateObservation(BaseModel):
    """Uma leitura de placa dentro do serviço (ainda não é o evento enviado ao backend)."""

    model_config = ConfigDict(frozen=True)

    camera_id: str = Field(min_length=1)
    plate_text: str = Field(min_length=1)  # texto como o OCR leu, ainda sem normalizar
    detection_confidence: float = Field(ge=0, le=1)  # confiança de ter achado uma placa
    ocr_confidence: float = Field(ge=0, le=1)  # confiança da leitura dos caracteres
    captured_at: AwareDatetime  # instante da captura do frame, em UTC (sem default de propósito)
    bbox: BoundingBox | None = None