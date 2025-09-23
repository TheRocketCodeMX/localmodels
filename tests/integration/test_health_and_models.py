import pytest

@pytest.mark.anyio
async def test_root_endpoint(http_client):
    resp = await http_client.get("/")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data.get("message"), "Root message missing"
    assert "endpoints" in data
    assert "available_models" in data

@pytest.mark.anyio
async def test_health_endpoint(http_client):
    resp = await http_client.get("/health")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data.get("status") in ("healthy", "unhealthy")
    assert data.get("ollama_service") in ("up", "down")

@pytest.mark.anyio
async def test_list_models_endpoint(http_client):
    resp = await http_client.get("/models")
    assert resp.status_code in (200, 500), resp.text  # 500 if Ollama not reachable
