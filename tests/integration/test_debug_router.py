import pytest


@pytest.mark.anyio
async def test_debug_router_invalid_model(http_client):
    resp = await http_client.post("/debug/litellm/no-existe-xyz", json={
        "message": "hola",
    })
    # Debug should return 404 for unknown model explicitly
    assert resp.status_code == 404, resp.text


@pytest.mark.anyio
async def test_debug_router_valid_model_smoke(http_client, available_models):
    # Try the first available model; this endpoint performs multiple probes
    model_key = available_models[0]
    resp = await http_client.post(f"/debug/litellm/{model_key}", json={
        "message": "hola",
        "max_tokens": 4,
    })
    # Allow upstream 5xx; on success ensure JSON has some expected keys
    assert resp.status_code in (200, 500), resp.text
    if resp.status_code == 200:
        data = resp.json()
        assert isinstance(data, dict)
        # Basic shape checks
        assert data.get("status") in ("ok", "warning", "error") or "result" in data
