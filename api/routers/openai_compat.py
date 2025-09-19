from typing import Dict, Any, Optional, List
import time
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

class OpenAIChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    max_tokens: int = 150
    temperature: float = 0.7
    stream: bool = False


def get_router(
    AVAILABLE_MODELS: Dict[str, Dict[str, Any]],
    ollama_base_url: str,
    logger,
    unified_completion,
) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/chat/completions")
    async def litellm_chat_completions(request: OpenAIChatRequest):
        try:
            if request.model not in AVAILABLE_MODELS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model {request.model} not available. Available models: {list(AVAILABLE_MODELS.keys())}",
                )

            default_reasoning = "medium"
            for msg in request.messages:
                if msg.get("role") == "system" and isinstance(msg.get("content"), str) and "Reasoning:" in msg["content"]:
                    try:
                        default_reasoning = msg["content"].split("Reasoning:")[1].strip().split(" ")[0]
                    except Exception:
                        pass
                    break

            completion_result = await unified_completion(
                model_name_key=request.model,
                messages=request.messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=request.stream,
                reasoning=default_reasoning,
                ollama_base_url=ollama_base_url,
            )

            if request.stream:
                async def generate_formatted_stream():
                    async for chunk in completion_result:
                        if hasattr(chunk, 'choices'):
                            yield f"data: {chunk.model_dump_json()}\n\n"
                        elif isinstance(chunk, dict) and "response" in chunk:
                            ollama_chunk = {
                                "id": "chatcmpl-ollama-fallback",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": request.model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": chunk["response"]},
                                        "finish_reason": "stop" if chunk.get("done") else None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(ollama_chunk)}\n\n"
                        if isinstance(chunk, dict) and chunk.get("done"):
                            yield "data: [DONE]\n\n"
                return StreamingResponse(generate_formatted_stream(), media_type="text/event-stream")
            else:
                if hasattr(completion_result, 'choices'):
                    return completion_result.model_dump()
                else:
                    ollama_response = {
                        "id": "chatcmpl-ollama-fallback",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": completion_result.get("response", "")},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": completion_result.get("prompt_eval_count", 0),
                            "completion_tokens": completion_result.get("eval_count", 0),
                            "total_tokens": completion_result.get("prompt_eval_count", 0) + completion_result.get("eval_eval_count", 0),
                        },
                    }
                    return ollama_response
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error in LiteLLM endpoint: {str(e)}")

    return router
