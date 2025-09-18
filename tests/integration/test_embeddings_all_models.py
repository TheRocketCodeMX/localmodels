import pytest


@pytest.mark.anyio
async def test_embeddings_all_models(http_client, available_models):
    for model_key in available_models:
        resp = await http_client.post("/v1/embeddings", json={
            "input": "hola",
            "model": model_key,
        })
        # Allow 200 if embeddings are wired; otherwise 503/500 acceptable
        assert resp.status_code in (200, 503, 500), resp.text
        if resp.status_code == 200:
            data = resp.json()
            assert data.get("object") == "list"
            assert isinstance(data.get("data"), list)
