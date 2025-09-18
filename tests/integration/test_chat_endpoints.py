import pytest

@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
async def test_chat_basic_all_models(http_client, available_models, stream):
    # Keep tokens and temperature low to make runs faster/stable
    for model_key in available_models:
        payload = {
            "message": "Di hola en una palabra",
            "model": model_key,
            "stream": stream,
            "max_tokens": 8,
            "temperature": 0.1,
        }
        if not stream:
            resp = await http_client.post("/chat", json=payload)
            assert resp.status_code == 200, f"/chat non-stream failed for {model_key}: {resp.text}"
            data = resp.json()
            assert data.get("model") == model_key
            assert isinstance(data.get("response"), str)
        else:
            # Stream: ensure we receive at least one line and [DONE]
            got_any = False
            got_done = False
            async with http_client.stream("POST", "/chat", json=payload) as resp:
                assert resp.status_code == 200, f"/chat stream failed for {model_key}: {await resp.aread()}"
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        got_any = True
                        if line.strip() == "data: [DONE]":
                            got_done = True
                            break
            assert got_any and got_done, f"Did not receive proper stream for {model_key}"
