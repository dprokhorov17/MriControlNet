"""Application configuration settings.

This module handles environment-based configuration for the MRI ControlNet application.
Settings are loaded from environment variables or .env file.
"""

from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """Application configuration settings.

    Attributes:
        APP_TITLE (str): The name of the application
        APP_VERSION (str): Application version number
        DEFAULT_CONTROLNET_MODEL (str): Path or name of the default ControlNet model
        DEFAULT_SD_MODEL (str): Path or name of the default Stable Diffusion model
        DEVICE (str): Computing device to use (e.g., 'cuda', 'cpu')
        DEFAULT_IMAGE_RESOLUTION (int): Default output image size
        DEFAULT_LOW_THRESHOLD (int): Default lower threshold for edge detection
        DEFAULT_HIGH_THRESHOLD (int): Default upper threshold for edge detection
        LOG_LEVEL (str): Logging verbosity level
        API_PREFIX (str): Prefix for all API endpoints
        MAX_FILE_SIZE_MB (int): Maximum allowed upload file size in MB
    """

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
