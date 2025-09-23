from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.config import settings, AVAILABLE_MODELS
from app.core.clients import (
    wait_for_ollama,
    setup_ollama_client,
    setup_litellm,
    setup_litellm_proxy_client,
    ensure_models_available,
)
from app.core.circuit_breaker import init_breakers


def lifespan_factory(logger):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Initialize circuit breakers
        init_breakers(list(AVAILABLE_MODELS.keys()))

        # Wait for Ollama
        if not await wait_for_ollama(settings.OLLAMA_BASE_URL, logger):
            raise RuntimeError("Could not connect to Ollama service")

        # Setup clients
        setup_ollama_client(settings.OLLAMA_BASE_URL, logger)
        setup_litellm(settings.OLLAMA_BASE_URL, logger)
        if settings.USE_LITELLM_PROXY:
            setup_litellm_proxy_client(settings.LITELLM_PROXY_URL, settings.LITELLM_MASTER_KEY, logger)

        # Ensure models
        await ensure_models_available(settings.OLLAMA_BASE_URL, AVAILABLE_MODELS, logger)

        logger.info("✅ Startup complete!")
        logger.info("🌐 Local API URL: http://localhost:8000")
        logger.info(f"🌐 LiteLLM Proxy: {settings.LITELLM_PROXY_URL} (enabled={settings.USE_LITELLM_PROXY})")
        logger.info("🌐 Web UI URL: http://localhost:3000")
        yield
        # Shutdown: no-op for now

    return lifespan
