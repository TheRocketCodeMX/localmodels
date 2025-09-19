import pytest


@pytest.mark.anyio
async def test_openai_stream_has_chunks(http_client, available_models):
    model_key = available_models[0]
    payload = {
        "model": model_key,
        "messages": [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Di hola"},
        ],
        "stream": True,
        "max_tokens": 8,
        "temperature": 0.1,
    }
    got_content = False
    got_done = False
    async with http_client.stream("POST", "/v1/chat/completions", json=payload) as resp:
        assert resp.status_code == 200, await resp.aread()
        async for line in resp.aiter_lines():
            if not line:
                continue
            if line.startswith("data: ") and line != "data: [DONE]":
                got_content = True
            if line.strip() == "data: [DONE]":
                got_done = True
                break
    assert got_content and got_done
