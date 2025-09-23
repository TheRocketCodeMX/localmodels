import os
import pytest
import httpx

# Base URL of the running gateway (FastAPI) service. Override in CI if needed.
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8000")
TIMEOUT = float(os.getenv("TEST_HTTP_TIMEOUT", "120"))

# Force AnyIO to use asyncio with session scope to avoid ScopeMismatch with session fixtures
@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"

@pytest.fixture(scope="session")
def base_url():
    return GATEWAY_URL.rstrip("/")

@pytest.fixture(scope="session")
async def http_client(base_url):
    async with httpx.AsyncClient(base_url=base_url, timeout=TIMEOUT) as client:
        yield client

@pytest.fixture(scope="session")
async def available_models(http_client):
    # Try to discover the available model keys from the root endpoint
    try:
        resp = await http_client.get("/")
        if resp.status_code == 200:
            data = resp.json()
            models = list(data.get("available_models", {}).keys())
            if models:
                return models
    except Exception:
        pass
    # Fallback: ask /health which returns a list of model keys
    try:
        resp = await http_client.get("/health")
        if resp.status_code == 200:
            data = resp.json()
            models = data.get("available_models")
            if isinstance(models, list) and models:
                return models
    except Exception:
        pass
    # Last resort: a minimal default list aligned with README/compose
    return [
        os.getenv("DEFAULT_MODEL", "qwen3:30b"),
    ]
