"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the FastAPI backend."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="FK_", case_sensitive=False)

    app_name: str = "Feynman-Kac PINN API"
    app_version: str = "0.1.0"
    app_description: str = "API for Feynman-Kac PINN simulations"

    api_v1_prefix: str = "/api/v1"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    default_training_steps: int = 25
    default_batch_size: int = 64
    default_mc_paths: int = 256
    default_learning_rate: float = 1e-3


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
