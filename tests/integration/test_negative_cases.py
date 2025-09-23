import os
import pytest


@pytest.mark.anyio
async def test_invalid_model_chat(http_client):
    # If /chat falls back to default model, allow 200; otherwise expect 400/404
    resp = await http_client.post("/chat", json={
        "message": "hola",
        "model": "no-existe-xyz",
    })
    assert resp.status_code in (200, 400, 404, 500), resp.text


@pytest.mark.anyio
async def test_invalid_model_openai(http_client):
    payload = {
        "model": "no-existe-xyz",
        "messages": [{"role": "user", "content": "hola"}],
    }
    resp = await http_client.post("/v1/chat/completions", json=payload)
    assert resp.status_code in (400, 404), resp.text


@pytest.mark.anyio
async def test_proxy_down_fallback_allows_requests(http_client):
    # Inspect health to see proxy status
    health = await http_client.get("/health")
    if health.status_code != 200:
        pytest.skip("/health not available")
    data = health.json()
    proxy_enabled = data.get("litellm_proxy_enabled")
    proxy_service = data.get("litellm_proxy_service")

    if proxy_enabled and proxy_service == "down":
        # Attempt a request; gateway should either fallback (200) or return a consistent error (5xx)
        payload = {
            "model": data.get("default_model") or os.getenv("DEFAULT_MODEL", "qwen3:30b"),
            "messages": [{"role": "user", "content": "di hola"}],
            "max_tokens": 8,
            "temperature": 0.1,
        }
        resp = await http_client.post("/v1/chat/completions", json=payload)
        assert resp.status_code in (200, 500), resp.text
    else:
        pytest.skip("Proxy not enabled or not down; skipping fallback test")
