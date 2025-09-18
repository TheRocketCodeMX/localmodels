import pytest


@pytest.mark.anyio
async def test_chat_direct_ollama(http_client, available_models):
    # Use first available model
    model_key = available_models[0]
    resp = await http_client.post("/chat/direct", json={
        "message": "di hola",
        "model": model_key,
        "max_tokens": 8,
        "temperature": 0.1,
    })
    # Allow upstream errors; ensure API responds (not hanging)
    assert resp.status_code in (200, 500), resp.text
    if resp.status_code == 200:
        data = resp.json()
        assert data.get("model") == model_key
        assert isinstance(data.get("response"), str)
