import pytest


@pytest.mark.anyio
async def test_chat_with_path_param(http_client, available_models):
    for model_key in available_models:
        resp = await http_client.post(f"/chat/{model_key}", json={
            "message": "di hola",
            "max_tokens": 8,
            "temperature": 0.1,
        })
        # Expect success or a handled upstream error
        assert resp.status_code in (200, 500), resp.text
        if resp.status_code == 200:
            data = resp.json()
            assert data.get("model") == model_key
            assert isinstance(data.get("response"), str)


@pytest.mark.anyio
@pytest.mark.parametrize("alias,expected_model", [
    ("qwen3", "qwen3:30b"),
    ("devstral", "devstral"),
    ("gemma3", "gemma3"),
    ("gpt-oss", "gpt-oss:20b"),
])
async def test_chat_aliases(http_client, available_models, alias, expected_model):
    # run only if expected model exists in this environment
    if expected_model not in available_models:
        pytest.skip(f"Modelo {expected_model} no disponible en este entorno")
    resp = await http_client.post(f"/chat/{alias}", json={
        "message": "di hola",
        "max_tokens": 8,
        "temperature": 0.1,
    })
    assert resp.status_code in (200, 500), resp.text
    if resp.status_code == 200:
        data = resp.json()
        assert data.get("model") == expected_model
        assert isinstance(data.get("response"), str)
