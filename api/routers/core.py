from typing import Dict, Any, Optional, List
import time
import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

# Local lightweight schemas to avoid importing from app.py
# and to keep routers decoupled.

def get_router(
    AVAILABLE_MODELS: Dict[str, Dict[str, Any]],
    default_model: str,
    ollama_base_url: str,
    use_litellm_proxy: bool,
    litellm_proxy_url: str,
    litellm_proxy_api_key: str,
    logger,
    unified_completion,  # callable passed in from app.py
) -> APIRouter:
    router = APIRouter()

    @router.get("/")
    async def root():
        return {
            "message": "Multi-Model API with LiteLLM is running",
            "default_model": default_model,
            "available_models": AVAILABLE_MODELS,
            "ollama_url": ollama_base_url,
            "litellm_proxy_enabled": use_litellm_proxy,
            "litellm_proxy_url": litellm_proxy_url,
            "endpoints": {
                "chat": "/chat",
                "chat_with_model": "/chat/{model_name}",
                "chat_litellm": "/v1/chat/completions",
                "embeddings": "/v1/embeddings",
                "health": "/health",
                "models": "/models",
            },
        }

    @router.get("/health")
    async def health():
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{ollama_base_url}/api/tags")
                ollama_healthy = response.status_code == 200

            proxy_healthy: Optional[bool] = None
            if use_litellm_proxy:
                try:
                    headers = {"Authorization": f"Bearer {litellm_proxy_api_key}"}
                    async with httpx.AsyncClient() as client:
                        proxy_resp = await client.get(f"{litellm_proxy_url}/v1/models", headers=headers, timeout=5)
                        proxy_healthy = proxy_resp.status_code == 200
                except Exception:
                    proxy_healthy = False

            return {
                "status": "healthy" if (ollama_healthy and (proxy_healthy in (None, True))) else "unhealthy",
                "ollama_service": "up" if ollama_healthy else "down",
                "litellm_proxy_enabled": use_litellm_proxy,
                "litellm_proxy_url": litellm_proxy_url,
                "litellm_proxy_service": ("up" if proxy_healthy else "down") if proxy_healthy is not None else "n/a",
                "default_model": default_model,
                "available_models": list(AVAILABLE_MODELS.keys()),
                "litellm_enabled": True,
            }
        except Exception as e:
            logger.error(f"Error in /health endpoint: {e}")
            return {
                "status": "unhealthy",
                "ollama_service": "down",
                "error": str(e),
            }

    @router.get("/models")
    async def list_models():
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{ollama_base_url}/api/tags")
                return response.json()
        except Exception as e:
            logger.error(f"Error fetching models: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")

    @router.get("/health/models")
    async def health_models():
        model_health_statuses: Dict[str, Any] = {}
        for model_key, model_info in AVAILABLE_MODELS.items():
            model_name = model_info["name"]
            status_entry: Dict[str, Any] = {"status": "unknown"}

            # 1. Check if model exists in Ollama
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(f"{ollama_base_url}/api/show", json={"name": model_name}, timeout=5)
                    if response.status_code == 200:
                        details = response.json()
                        status_entry.update({
                            "ollama_status": "available",
                            "size": details.get("size"),
                            "quantization": details.get("details", {}).get("quantization_level"),
                            "family": details.get("details", {}).get("family"),
                            "description": model_info.get("description"),
                        })
                    elif response.status_code == 404:
                        status_entry.update({"ollama_status": "not_found", "error": "Model not found in Ollama"})
                    else:
                        status_entry.update({"ollama_status": "api_error", "error": f"Ollama API returned status {response.status_code}"})
            except httpx.RequestError as e:
                status_entry.update({"ollama_status": "network_error", "error": f"Network or request error: {str(e)}"})
            except Exception as e:
                status_entry.update({"ollama_status": "unexpected_error", "error": f"Unexpected error during Ollama check: {str(e)}"})

            # 2. Perform a quick LiteLLM completion test for responsiveness and latency
            test_latency_ms = None
            test_success = False
            if status_entry.get("ollama_status") == "available":
                try:
                    test_messages = [{"role": "user", "content": "Hi"}]
                    test_start_time = time.time()
                    test_response = await unified_completion(
                        model_name_key=model_key,
                        messages=test_messages,
                        max_tokens=5,
                        temperature=0.1,
                        stream=False,
                        reasoning="none",
                        ollama_base_url=ollama_base_url,
                    )
                    test_end_time = time.time()
                    test_latency_ms = (test_end_time - test_start_time) * 1000

                    # Consider LiteLLM responsive even if content comes back empty for some models.
                    # Try to determine success more robustly and fall back to a direct Ollama probe when needed.
                    if hasattr(test_response, 'choices'):
                        try:
                            content = test_response.choices[0].message.content or ""
                            content = content.strip() if isinstance(content, str) else content
                        except Exception:
                            content = ""
                        if content:
                            test_success = True
                        else:
                            # Some models (e.g., gpt-oss:20b) occasionally return empty content via LiteLLM
                            # Attempt a very small direct Ollama generate to verify model health
                            try:
                                async with httpx.AsyncClient(timeout=10.0) as client:
                                    payload = {
                                        "model": model_name,
                                        "prompt": "Say ok.",
                                        "options": {"num_predict": 4, "temperature": 0.1},
                                        "stream": False,
                                    }
                                    direct = await client.post(f"{ollama_base_url}/api/generate", json=payload)
                                    if direct.status_code == 200:
                                        dj = direct.json()
                                        direct_resp = (dj.get("response") or "").strip()
                                        if direct_resp:
                                            test_success = True
                                            status_entry["litellm_test_note"] = "empty_content_but_direct_ok"
                            except Exception:
                                # Ignore fallback errors here; we'll keep test_success as False
                                pass
                    elif isinstance(test_response, dict) and test_response.get("response"):
                        test_success = True

                    status_entry.update({
                        "litellm_test_status": "responsive",
                        "litellm_test_latency_ms": round(test_latency_ms, 2),
                        "litellm_test_success": test_success,
                    })
                except Exception as e:
                    status_entry.update({
                        "litellm_test_status": "unresponsive",
                        "litellm_test_error": str(e),
                        "litellm_test_latency_ms": test_latency_ms,
                        "litellm_test_success": False,
                    })

            status_entry["status"] = (
                "healthy" if (status_entry.get("ollama_status") == "available" and status_entry.get("litellm_test_success")) else "unhealthy"
            )
            model_health_statuses[model_key] = status_entry

        return JSONResponse(content=model_health_statuses)

    return router
