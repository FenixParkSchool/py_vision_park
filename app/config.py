from functools import lru_cache

from pydantic import Field

# BaseSettings é uma classe pronta que sabe ler variáveis de ambiente e do arquivo .env
from pydantic_settings import BaseSettings, SettingsConfigDict

# herdando tudo do BaseSettings
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    # “leia o arquivo .env” e “ignore variáveis que não estão na classe”

    camera_source: str = "data/teste.mp4"
    camera_id: str = "portaria-1"
    backend_url: str = "http://localhost:3000"
    backend_api_key: str = ""
    min_confidence: float = Field(default=0.6, ge=0, le=1)
    dedup_seconds: float = Field(default=30, gt=0)
    sample_interval_seconds: float = Field(default=1.0, gt=0)
    
    # Os valores 0.6, 30 e 1.0 são padrões de desenvolvimento, ainda não validados. Só serão ajustados quando houver dados reais.


@lru_cache
def get_settings() -> Settings:
    return Settings()