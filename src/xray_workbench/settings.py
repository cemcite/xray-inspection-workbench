from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="XRAY_", env_file=".env", extra="ignore")

    application_name: str = "X-Ray Inspection Workbench"
    database_url: str = "sqlite:///./xray_workbench.db"
    image_storage_dir: str = "data/runtime/images"
    risk_config: str = "config/risk-thresholds.yaml"
    model_name: str = "xray-detector"
    model_version: str = "unavailable"
    model_path: str | None = None
    model_min_confidence: float = 0.10


@lru_cache
def get_settings() -> Settings:
    return Settings()
