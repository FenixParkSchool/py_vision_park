from functools import lru_cache
from typing import Literal

from pydantic import Field

# BaseSettings é uma classe pronta que sabe ler variáveis de ambiente e do arquivo .env
from pydantic_settings import BaseSettings, SettingsConfigDict

# herdando tudo do BaseSettings
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_ignore_empty=True)
    # “leia o arquivo .env” e “ignore variáveis que não estão na classe”

    camera_source: str = "data/teste.mp4"
    camera_id: str = "portaria-1"
    backend_url: str = "http://localhost:3000"
    backend_api_key: str = ""
    min_confidence: float = Field(default=0.6, ge=0, le=1)
    dedup_seconds: float = Field(default=30, gt=0)
    sample_interval_seconds: float = Field(default=1.0, gt=0)
    backend_timeout: float = Field(default=5, gt=0)
    backend_max_retries: int = Field(default=3, ge=0)
    # "fake" = detector/OCR de mentira (desenvolvimento); "tf" = modelos Keras treinados;
    # "fpo" = nosso detector + OCR pronto fast-plate-ocr (recomendado).
    recognition_mode: Literal["fake", "tf", "fpo"] = "fake"
    fpo_model: str = "cct-s-v2-global-model"
    detector_model_path: str = "models/detector.keras"
    ocr_model_path: str = "models/ocr.keras"
    detector_threshold: float = Field(default=0.5, ge=0, le=1)  # limiar do mapa de probabilidade
    # Se vazio, o OCR usa o mesmo min_confidence da detecção. Separar só com dados reais.
    ocr_min_confidence: float | None = Field(default=None, ge=0, le=1)
    # Votação entre quadros (app/services/voter.py), em QUADROS AMOSTRADOS (ver
    # sample_interval_seconds): envia o veículo após vote_max_reads leituras ou após
    # vote_gap_frames quadros sem vê-lo. Com amostragem de 1 s, 3 quadros = ~3 s. Provisório.
    vote_max_reads: int = Field(default=3, ge=1)
    vote_gap_frames: int = Field(default=3, ge=1)

    # Os valores 0.6, 30 e 1.0 são padrões de desenvolvimento, ainda não validados. Só serão ajustados quando houver dados reais.


@lru_cache
def get_settings() -> Settings:
    return Settings()