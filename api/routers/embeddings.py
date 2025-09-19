from typing import Dict, Any, List, Union, Callable
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]

class EmbeddingData(BaseModel):
    index: int
    embedding: List[float]

class EmbeddingResponse(BaseModel):
    data: List[EmbeddingData]
    model: str
    usage: Dict[str, int]


def get_router(
    AVAILABLE_MODELS: Dict[str, Dict[str, Any]],
    default_model: str,
    ollama_base_url: str,
    logger,
    is_ollama_client_ready: Callable[[], bool],
) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/embeddings", response_model=EmbeddingResponse)
    async def create_embeddings(request: EmbeddingRequest):
        if not is_ollama_client_ready():
            raise HTTPException(status_code=503, detail="Ollama client not initialized")
        try:
            texts = [request.input] if isinstance(request.input, str) else request.input
            embedding_data: List[EmbeddingData] = []
            total_tokens = 0

            for i, text in enumerate(texts):
                async with httpx.AsyncClient(timeout=300.0) as client:
                    embedding_model = request.model if request.model and request.model in AVAILABLE_MODELS else default_model
                    ollama_request = {"model": embedding_model, "prompt": text}
                    response = await client.post(f"{ollama_base_url}/api/embeddings", json=ollama_request)
                    if response.status_code == 200:
                        result = response.json()
                        embedding = result.get("embedding", []) or [0.0] * 1536
                        embedding_data.append(EmbeddingData(index=i, embedding=embedding))
                        total_tokens += len(text.split())
                    else:
                        logger.warning(f"⚠️ Ollama embeddings not available, using dummy embedding for: {text[:50]}...")
                        embedding_data.append(EmbeddingData(index=i, embedding=[0.0] * 1536))
                        total_tokens += len(text.split())

            usage_info = {"prompt_tokens": total_tokens, "total_tokens": total_tokens}
            return EmbeddingResponse(data=embedding_data, model=request.model or default_model, usage=usage_info)
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

    return router
