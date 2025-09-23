from typing import Dict, Any, Optional, List
import time
import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

class ChatRequest(BaseModel):
    model: Optional[str] = None
    message: str
    max_tokens: int = 150
    temperature: float = 0.7
    stream: bool = False
    reasoning: str = ""

class ChatResponse(BaseModel):
    response: str
    model: str
    usage: Dict[str, int] = {}


def get_router(
    AVAILABLE_MODELS: Dict[str, Dict[str, Any]],
    default_model: str,
    ollama_base_url: str,
    logger,
    unified_completion,
) -> APIRouter:
    router = APIRouter()

    @router.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        selected_model = request.model if request.model and request.model in AVAILABLE_MODELS else default_model

        messages_for_completion = [
            {"role": "system", "content": f"You are a helpful assistant. Reasoning: {request.reasoning}"},
            {"role": "user", "content": request.message},
        ]

        try:
            completion_result = await unified_completion(
                model_name_key=selected_model,
                messages=messages_for_completion,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=request.stream,
                reasoning=request.reasoning,
                ollama_base_url=ollama_base_url,
            )

            if request.stream:
                async def generate_formatted_stream():
                    async for chunk in completion_result:
                        if hasattr(chunk, 'choices') and chunk.choices[0].delta.content:
                            yield f"data: {chunk.choices[0].delta.content}\n\n"
                        elif isinstance(chunk, dict) and "response" in chunk:
                            yield f"data: {chunk['response']}\n\n"
                        if (hasattr(chunk, 'choices') and chunk.choices[0].finish_reason) or (isinstance(chunk, dict) and chunk.get("done")):
                            yield "data: [DONE]\n\n"
                return StreamingResponse(generate_formatted_stream(), media_type="text/plain")
            else:
                if hasattr(completion_result, 'choices'):
                    content = completion_result.choices[0].message.content
                    usage_info = {
                        "prompt_tokens": getattr(completion_result.usage, 'prompt_tokens', 0),
                        "completion_tokens": getattr(completion_result.usage, 'completion_tokens', 0),
                        "total_tokens": getattr(completion_result.usage, 'total_tokens', 0),
                    } if hasattr(completion_result, 'usage') and completion_result.usage else {}
                else:
                    content = completion_result.get("response", "")
                    usage_info = {
                        "prompt_tokens": completion_result.get("prompt_eval_count", 0),
                        "completion_tokens": completion_result.get("eval_count", 0),
                        "total_tokens": completion_result.get("prompt_eval_count", 0) + completion_result.get("eval_eval_count", 0),
                    }

                return ChatResponse(response=content, model=selected_model, usage=usage_info)
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating response with model {selected_model}: {str(e)}")

    @router.post("/chat/direct", response_model=ChatResponse)
    async def chat_direct_ollama(request: ChatRequest):
        selected_model = request.model if request.model and request.model in AVAILABLE_MODELS else default_model
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                ollama_request = {
                    "model": selected_model,
                    "prompt": f"Reasoning: {request.reasoning}\nUser: {request.message}\nAssistant:",
                    "options": {"num_predict": request.max_tokens, "temperature": request.temperature},
                    "stream": False,
                }
                response = await client.post(f"{ollama_base_url}/api/generate", json=ollama_request)
                if response.status_code == 200:
                    result = response.json()
                    return ChatResponse(
                        response=result.get("response", ""),
                        model=selected_model,
                        usage={
                            "prompt_tokens": result.get("prompt_eval_count", 0),
                            "completion_tokens": result.get("eval_count", 0),
                            "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_eval_count", 0),
                        },
                    )
                else:
                    logger.error(f"Ollama API error: {response.status_code}")
                    raise HTTPException(status_code=500, detail=f"Ollama API error: {response.status_code}")
        except Exception as e:
            logger.error(f"Error with direct Ollama call: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error with direct Ollama call: {str(e)}")

    @router.post("/chat/{model_name}", response_model=ChatResponse)
    async def chat_with_specific_model(model_name: str, request: ChatRequest):
        # Support common aliases mapping to actual model keys
        alias_map = {
            "qwen3": "qwen3:30b",
            "gpt-oss": "gpt-oss:20b",
        }
        resolved = model_name
        if resolved not in AVAILABLE_MODELS:
            resolved = alias_map.get(model_name, resolved)
        if resolved not in AVAILABLE_MODELS:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        request.model = resolved
        return await chat(request)

    # Convenience aliases mirroring the original endpoints
    @router.post("/chat/qwen3", response_model=ChatResponse)
    async def chat_qwen3(request: ChatRequest):
        request.model = "qwen3:30b"
        return await chat(request)

    @router.post("/chat/devstral", response_model=ChatResponse)
    async def chat_devstral(request: ChatRequest):
        request.model = "devstral"
        return await chat(request)

    @router.post("/chat/gemma3", response_model=ChatResponse)
    async def chat_gemma3(request: ChatRequest):
        request.model = "gemma3"
        return await chat(request)

    @router.post("/chat/gpt-oss", response_model=ChatResponse)
    async def chat_gpt_oss(request: ChatRequest):
        request.model = "gpt-oss:20b"
        return await chat(request)

    return router
