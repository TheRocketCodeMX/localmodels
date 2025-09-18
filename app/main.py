import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from logging_config import setup_logging

from app.core.config import settings, AVAILABLE_MODELS
from app.core.lifecycle import lifespan_factory
from app.core.clients import ollama_client
from app.services.completion import unified_completion as _unified_completion

from api.routers import core as core_router
from api.routers import chat as chat_router
from api.routers import openai_compat as openai_router
from api.routers import embeddings as embeddings_router
from api.routers import debug as debug_router


logger = setup_logging()

# Build lifespan with logger context
lifespan = lifespan_factory(logger)

app = FastAPI(title="Multi-Model API with LiteLLM", version="6.0.0", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _unified_completion_adapter(**kwargs):
    # Keep router signature the same but pass config values here
    return _unified_completion(
        use_litellm_proxy=settings.USE_LITELLM_PROXY,
        logger=logger,
        **kwargs,
    )


def _is_ollama_ready() -> bool:
    try:
        return ollama_client is not None
    except Exception:
        return False


# Include routers
app.include_router(core_router.get_router(
    AVAILABLE_MODELS,
    settings.DEFAULT_MODEL,
    settings.OLLAMA_BASE_URL,
    settings.USE_LITELLM_PROXY,
    settings.LITELLM_PROXY_URL,
    settings.LITELLM_MASTER_KEY,
    logger,
    _unified_completion_adapter,
))

app.include_router(chat_router.get_router(
    AVAILABLE_MODELS,
    settings.DEFAULT_MODEL,
    settings.OLLAMA_BASE_URL,
    logger,
    _unified_completion_adapter,
))

app.include_router(openai_router.get_router(
    AVAILABLE_MODELS,
    settings.OLLAMA_BASE_URL,
    logger,
    _unified_completion_adapter,
))

app.include_router(embeddings_router.get_router(
    AVAILABLE_MODELS,
    settings.DEFAULT_MODEL,
    settings.OLLAMA_BASE_URL,
    logger,
    _is_ollama_ready,
))

# Debug router
app.include_router(debug_router.get_router(
    AVAILABLE_MODELS,
    settings.OLLAMA_BASE_URL,
    logger,
))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
