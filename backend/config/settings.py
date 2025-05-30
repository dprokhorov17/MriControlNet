from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    APP_TITLE: str
    APP_VERSION: str
    DEFAULT_CONTROLNET_MODEL: str
    DEFAULT_SD_MODEL: str
    DEVICE: str
    DEFAULT_IMAGE_RESOLUTION: int
    DEFAULT_LOW_THRESHOLD: int
    DEFAULT_HIGH_THRESHOLD: int
    LOG_LEVEL: str
    API_PREFIX: str = "/api/v1"
    MAX_FILE_SIZE_MB: int = 10

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
