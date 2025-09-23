import os
import time
import json
import httpx
from fastapi import HTTPException
from typing import List, Dict
from litellm import acompletion

from app.core.clients import openai_proxy_client
from app.core.circuit_breaker import get_breaker
from app.core.config import AVAILABLE_MODELS


async def unified_completion(
    model_name_key: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    stream: bool,
    reasoning: str,
    ollama_base_url: str,
    use_litellm_proxy: bool,
    logger,
):
    selected_model_info = AVAILABLE_MODELS[model_name_key]
    litellm_model = selected_model_info["litellm_name"]
    ollama_model_name = selected_model_info["name"]

    litellm_timeout = int(os.getenv("LITELLM_TIMEOUT", 600))
    litellm_retries = int(os.getenv("LITELLM_RETRIES", 3))

    start_time = time.time()
    success = False

    breaker = get_breaker(model_name_key)

    try:
        if stream:
            async def generate_stream():
                nonlocal success
                breaker.begin()
                try:
                    if use_litellm_proxy:
                        if openai_proxy_client is None:
                            raise RuntimeError("LiteLLM proxy client not initialized")
                        stream_resp = await openai_proxy_client.chat.completions.create(
                            model=model_name_key,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            stream=True,
                        )
                        async for chunk in stream_resp:
                            yield chunk
                        success = True
                    else:
                        gen = await acompletion(
                            model=litellm_model,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            stream=True,
                            api_base=ollama_base_url,
                            timeout=litellm_timeout,
                            num_retries=litellm_retries,
                        )
                        async for chunk in gen:
                            yield chunk
                        success = True
                except Exception as primary_error:
                    try:
                        async with httpx.AsyncClient(timeout=litellm_timeout) as client:
                            ollama_prompt = f"Reasoning: {reasoning}\nUser: {messages[-1]['content']}\nAssistant:"
                            payload = {
                                "model": ollama_model_name,
                                "prompt": ollama_prompt,
                                "options": {"num_predict": max_tokens, "temperature": temperature},
                                "stream": True,
                            }
                            async with client.stream("POST", f"{ollama_base_url}/api/generate", json=payload) as resp:
                                if resp.status_code == 200:
                                    async for line in resp.aiter_lines():
                                        if not line:
                                            continue
                                        try:
                                            data = json.loads(line)
                                            yield data
                                            if data.get("done"):
                                                success = True
                                        except json.JSONDecodeError:
                                            continue
                                else:
                                    raise HTTPException(status_code=500, detail=f"Direct Ollama stream failed {resp.status_code}")
                    except Exception as fallback_error:
                        raise HTTPException(status_code=500, detail=f"All streaming methods failed: {primary_error} | {fallback_error}")
                finally:
                    latency = (time.time() - start_time) * 1000
                    if success:
                        breaker.success()
                    else:
                        breaker.failure()
                    logger.info(f"METRIC: Model={model_name_key}, Type=Streaming, Latency={latency:.2f}ms, Success={success}")
            return generate_stream()
        else:
            breaker.begin()
            try:
                if use_litellm_proxy:
                    if openai_proxy_client is None:
                        raise RuntimeError("LiteLLM proxy client not initialized")
                    resp = await openai_proxy_client.chat.completions.create(
                        model=model_name_key,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=False,
                    )
                else:
                    resp = await acompletion(
                        model=litellm_model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        api_base=ollama_base_url,
                        timeout=litellm_timeout,
                        num_retries=litellm_retries,
                    )
                success = True
                breaker.success()
                return resp
            except Exception as primary_error:
                try:
                    async with httpx.AsyncClient(timeout=litellm_timeout) as client:
                        ollama_prompt = f"Reasoning: {reasoning}\nUser: {messages[-1]['content']}\nAssistant:"
                        payload = {
                            "model": ollama_model_name,
                            "prompt": ollama_prompt,
                            "options": {"num_predict": max_tokens, "temperature": temperature},
                            "stream": False,
                        }
                        direct = await client.post(f"{ollama_base_url}/api/generate", json=payload)
                        if direct.status_code == 200:
                            success = True
                            breaker.success()
                            return direct.json()
                        else:
                            breaker.failure()
                            raise HTTPException(status_code=500, detail=f"Direct Ollama failed {direct.status_code}")
                except Exception as fallback_error:
                    breaker.failure(fallback_error)
                    raise HTTPException(status_code=500, detail=f"All methods failed: {primary_error} | {fallback_error}")
            finally:
                latency = (time.time() - start_time) * 1000
                logger.info(f"METRIC: Model={model_name_key}, Type=Non-Streaming, Latency={latency:.2f}ms, Success={success}")
    except Exception:
        raise
