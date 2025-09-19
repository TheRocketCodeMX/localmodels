import pytest


@pytest.mark.anyio
async def test_health_models(http_client, available_models):
    resp = await http_client.get("/health/models")
    # Health models may return 200 with detailed JSON or 500 if upstream not reachable
    assert resp.status_code in (200, 500), resp.text
    if resp.status_code == 200:
        data = resp.json()
        # Should have at least an entry per configured model
        for mk in available_models:
            assert mk in data, f"Falta estado para {mk}"
            entry = data[mk]
            assert entry.get("ollama_status") in (
                "available",
                "not_found",
                "api_error",
                "network_error",
                "unexpected_error",
            )
