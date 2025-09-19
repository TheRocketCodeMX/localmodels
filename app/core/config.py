from typing import Dict, Any
from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    USE_LITELLM_PROXY: bool = os.getenv("USE_LITELLM_PROXY", "true").lower() in ("1", "true", "yes", "on")
    LITELLM_PROXY_URL: str = os.getenv("LITELLM_PROXY_URL", "http://litellm-proxy:4000")
    LITELLM_MASTER_KEY: str = os.getenv("LITELLM_MASTER_KEY", "sk-1234")
    # Align default with plan: prefer qwen3:30b when no env variable is set
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "qwen3:30b")

    model_config = {
        "env_file": ".env",
        "extra": "ignore",
    }


# Available models configuration (moved from app.py)
AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
    "gpt-oss:20b": {
        "name": "gpt-oss:20b",
        "description": "GPT-OSS 20B model",
        "provider": "ollama",
        "litellm_name": "ollama/gpt-oss:20b",
    },
    "qwen3:30b": {
        "name": "qwen3:30b",
        "description": "Qwen 3.0 30B model",
        "provider": "ollama",
        "litellm_name": "ollama/qwen3:30b",
    },
    "devstral": {
        "name": "devstral",
        "description": "DevStral model",
        "provider": "ollama",
        "litellm_name": "ollama/devstral",
    },
    "gemma3": {
        "name": "gemma3",
        "description": "Gemma 3 model",
        "provider": "ollama",
        "litellm_name": "ollama/gemma3",
    },
}


settings = Settings()
