import os
import json
import asyncio
import httpx
from openai import OpenAI, AsyncOpenAI
from typing import Optional, Dict, Any

# Globals for clients
ollama_client: Optional[OpenAI] = None
openai_proxy_client: Optional[AsyncOpenAI] = None


async def wait_for_ollama(base_url: str, logger, retries: int = 30) -> bool:
    retry_count = 0
    while retry_count < retries:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{base_url}/api/tags")
                if response.status_code == 200:
                    logger.info("✅ Ollama service is ready")
                    return True
        except Exception:
            logger.warning(f"⏳ Waiting for Ollama service... ({retry_count + 1}/{retries})")
        await asyncio.sleep(2)
        retry_count += 1
    logger.error("❌ Could not connect to Ollama service")
    return False


async def ensure_models_available(base_url: str, available_models: Dict[str, Dict[str, Any]], logger) -> bool:
    try:
        async with httpx.AsyncClient(timeout=1800.0) as client:
            response = await client.get(f"{base_url}/api/tags")
            existing_models = response.json().get("models", [])
            existing_model_names = [model.get("name", "") for model in existing_models]

            for _, model_info in available_models.items():
                model_name = model_info["name"]
                model_exists = any(model_name in existing for existing in existing_model_names)
                if not model_exists:
                    logger.info(f"📥 Downloading model {model_name}... This may take 10-30 minutes.")
                    logger.info("💡 Large models may require significant disk space and RAM")
                    pull_data = {"name": model_name, "stream": True}
                    async with client.stream("POST", f"{base_url}/api/pull", json=pull_data) as pull_response:
                        if pull_response.status_code == 200:
                            async for line in pull_response.aiter_lines():
                                if not line:
                                    continue
                                try:
                                    data = json.loads(line)
                                    status = data.get("status") or data.get("status_message")
                                    if status:
                                        logger.info(f"📡 {model_name}: {status}")
                                    if data.get("status") == "success":
                                        logger.info(f"✅ Model {model_name} downloaded successfully!")
                                        break
                                except Exception:
                                    continue
                        else:
                            logger.warning(f"⚠️ Failed to download model {model_name}. Status: {pull_response.status_code}")
                else:
                    logger.info(f"✅ Model {model_name} already available")
        return True
    except Exception as e:
        logger.error(f"❌ Error ensuring model availability: {e}")
        return True


def setup_ollama_client(base_url: str, logger):
    global ollama_client
    ollama_client = OpenAI(base_url=f"{base_url}/v1", api_key="ollama")
    logger.info("✅ Ollama client configured")


def setup_litellm(base_url: str, logger):
    import litellm
    # Verbose logging to aid debugging and observability
    litellm.set_verbose = True
    os.environ["OLLAMA_API_BASE"] = base_url
    os.environ["OLLAMA_API_KEY"] = "ollama"
    os.environ["LITELLM_LOG"] = "DEBUG"
    litellm.drop_params = True
    litellm.max_budget = 100
    if os.getenv("ENABLE_LANGFUSE", "false").lower() in ("1", "true", "yes", "on"):
        litellm.success_callback = ["langfuse"]
        litellm.failure_callback = ["langfuse"]
    logger.info("✅ LiteLLM configured for Ollama")


def setup_litellm_proxy_client(base_url: str, api_key: str, logger):
    global openai_proxy_client
    try:
        openai_proxy_client = AsyncOpenAI(base_url=f"{base_url}/v1", api_key=api_key)
        logger.info("✅ LiteLLM proxy client configured")
    except Exception as e:
        logger.error(f"❌ Failed to configure LiteLLM proxy client: {e}")
        openai_proxy_client = None
