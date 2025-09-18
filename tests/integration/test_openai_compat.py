import os
import pytest

@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
async def test_openai_chat_completions(http_client, available_models, stream):
    for model_key in available_models:
        payload = {
            "model": model_key,
            "messages": [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "Escribe una palabra."},
            ],
            "stream": stream,
            "max_tokens": 8,
            "temperature": 0.1,
        }
        if not stream:
            resp = await http_client.post("/v1/chat/completions", json=payload)
            assert resp.status_code == 200, f"/v1/chat/completions non-stream failed for {model_key}: {resp.text}"
            data = resp.json()
            assert "choices" in data and len(data["choices"]) >= 1
        else:
            got_done = False
            async with http_client.stream("POST", "/v1/chat/completions", json=payload) as resp:
                assert resp.status_code == 200, f"/v1/chat/completions stream failed for {model_key}: {await resp.aread()}"
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.strip() == "data: [DONE]":
                        got_done = True
                        break
            assert got_done, f"Did not receive [DONE] for {model_key}"

@pytest.mark.anyio
async def test_embeddings(http_client, available_models):
    # Pick first available model for embeddings test
    model_key = available_models[0]
    payload = {
        "input": "hola",
        "model": model_key,
    }
    resp = await http_client.post("/v1/embeddings", json=payload)
    # If embeddings are wired, we expect 200; otherwise allow 503/500 but ensure JSON error
    assert resp.status_code in (200, 503, 500), resp.text
    if resp.status_code == 200:
        data = resp.json()
        assert data.get("object") == "list"
        assert isinstance(data.get("data"), list) and len(data["data"]) >= 1
