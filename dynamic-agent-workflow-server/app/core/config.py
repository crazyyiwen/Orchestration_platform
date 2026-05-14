from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- App ---
    APP_NAME: str = "dynamic-agent-workflow-server"
    APP_ENV: Literal["local", "dev", "staging", "prod"] = "local"
    API_PREFIX: str = "/api"
    # ``NoDecode`` tells pydantic-settings to NOT JSON-parse this field;
    # the comma-split happens in the ``_split_origins`` validator below.
    FRONTEND_ORIGINS: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["*"]
    )
    LOG_LEVEL: str = "INFO"

    # --- MongoDB ---
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DATABASE: str = "agent_workflow_runtime"

    # --- Metadata API (optional source for workflow definitions) ---
    METADATA_API_ENABLED: bool = False
    METADATA_API_BASE_URL: str = "http://localhost:8001"

    # --- Langfuse ---
    LANGFUSE_ENABLED: bool = False
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

    # --- LLM providers ---
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    HUGGINGFACE_API_KEY: str = ""
    DEFAULT_MODEL_ID: str = "mock-fast"

    # --- Runtime limits & gates ---
    ENABLE_SCRIPT_NODE: bool = False
    ALLOW_EXTERNAL_HTTP: bool = True
    MAX_WORKFLOW_STEPS: int = 100
    MAX_AGENT_ITERATIONS: int = 8
    MAX_SUBFLOW_DEPTH: int = 3
    NODE_TIMEOUT_SECONDS: int = 60
    WORKFLOW_TIMEOUT_SECONDS: int = 600

    @field_validator("FRONTEND_ORIGINS", mode="before")
    @classmethod
    def _split_origins(cls, v: object) -> object:
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
