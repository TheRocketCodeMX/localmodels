import os
import asyncio
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
import uvicorn
import threading
import time

app = FastAPI(title="GPT-OSS API via Ollama", version="1.0.0")

# Ollama client
ollama_client = None
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Available models configuration
AVAILABLE_MODELS = {
    "gpt-oss:20b": {"name": "gpt-oss:20b", "description": "GPT-OSS 20B model"},
    "qwen3:30b": {"name": "qwen3:30b", "description": "Qwen 3.0 30B model"},
    "devstral": {"name": "devstral", "description": "DevStral model"}, 
    "gemma3": {"name": "gemma3", "description": "Gemma 3 model"}
}

# Default model
default_model = os.getenv("DEFAULT_MODEL", "gpt-oss:20b")

class ChatRequest(BaseModel):
    message: str
    model: str = None
    max_tokens: int = 512
    temperature: float = 0.7
    reasoning: str = "medium"
    stream: bool = False

class ChatResponse(BaseModel):
    response: str
    model: str
    usage: dict = None

class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str = None
    encoding_format: str = "float"
    dimensions: int = None

class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: list[float]

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: dict

async def wait_for_ollama():
    """Wait for Ollama service to be ready"""
    max_retries = 30
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{ollama_base_url}/api/tags")
                if response.status_code == 200:
                    print("✅ Ollama service is ready")
                    return True
        except Exception as e:
            print(f"⏳ Waiting for Ollama service... ({retry_count + 1}/{max_retries})")
            await asyncio.sleep(2)
            retry_count += 1
    
    print("❌ Could not connect to Ollama service")
    return False

async def ensure_models_available():
    """Download all configured models if not available"""
    try:
        # Use longer timeout for model download (30 minutes)
        async with httpx.AsyncClient(timeout=1800.0) as client:
            # Check if models are already available
            response = await client.get(f"{ollama_base_url}/api/tags")
            existing_models = response.json().get("models", [])
            existing_model_names = [model["name"] for model in existing_models]
            
            for model_key, model_info in AVAILABLE_MODELS.items():
                model_name = model_info["name"]
                
                model_exists = any(model_name in existing for existing in existing_model_names)
                
                if not model_exists:
                    print(f"📥 Downloading model {model_name}... This may take 10-30 minutes.")
                    print(f"💡 Large models may require significant disk space and RAM")
                    
                    # Pull the model with streaming to show progress
                    pull_data = {"name": model_name, "stream": True}
                    
                    async with client.stream("POST", f"{ollama_base_url}/api/pull", json=pull_data) as pull_response:
                        if pull_response.status_code == 200:
                            async for line in pull_response.aiter_lines():
                                if line:
                                    try:
                                        data = eval(line)  # Parse JSON line
                                        if "status" in data:
                                            print(f"📡 {model_name}: {data['status']}")
                                        if data.get("status") == "success":
                                            print(f"✅ Model {model_name} downloaded successfully!")
                                            break
                                    except:
                                        continue
                        else:
                            print(f"⚠️ Failed to download model {model_name}. Status: {pull_response.status_code}")
                            print(f"   Model will be available for manual download: docker exec -it <container> ollama pull {model_name}")
                else:
                    print(f"✅ Model {model_name} already available")
            
            return True
                
    except Exception as e:
        print(f"❌ Error ensuring model availability: {e}")
        print(f"💡 You can manually download models with: docker exec -it <container> ollama pull <model_name>")
        return True  # Return True to continue startup

def setup_ollama_client():
    """Setup OpenAI client pointing to Ollama"""
    global ollama_client
    
    ollama_client = OpenAI(
        base_url=f"{ollama_base_url}/v1",
        api_key="ollama"  # dummy key for Ollama
    )
    print("✅ Ollama client configured")

@app.on_event("startup")
async def startup_event():
    print("🚀 Starting GPT-OSS API Gateway...")
    
    # Wait for Ollama service
    if not await wait_for_ollama():
        raise RuntimeError("Could not connect to Ollama service")
    
    # Setup client
    setup_ollama_client()
    
    # Ensure models are available
    await ensure_models_available()
    
    print("✅ Startup complete!")

@app.get("/")
async def root():
    return {
        "message": "Multi-Model API via Ollama is running",
        "default_model": default_model,
        "available_models": AVAILABLE_MODELS,
        "ollama_url": ollama_base_url,
        "endpoints": {
            "chat": "/chat",
            "chat_with_model": "/chat/{model_name}",
            "embeddings": "/v1/embeddings",
            "health": "/health",
            "models": "/models",
            "ollama_direct": f"{ollama_base_url}/v1/chat/completions"
        }
    }

@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{ollama_base_url}/api/tags")
            ollama_healthy = response.status_code == 200
            
        return {
            "status": "healthy" if ollama_healthy else "unhealthy",
            "ollama_service": "up" if ollama_healthy else "down",
            "default_model": default_model,
            "available_models": list(AVAILABLE_MODELS.keys())
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "ollama_service": "down",
            "error": str(e)
        }

@app.get("/models")
async def list_models():
    """List available models in Ollama"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{ollama_base_url}/api/tags")
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if ollama_client is None:
        raise HTTPException(status_code=503, detail="Ollama client not initialized")
    
    # Select model - use request model if provided, otherwise default
    selected_model = request.model if request.model and request.model in AVAILABLE_MODELS else default_model
    
    try:
        # Format message with reasoning level
        system_content = f"You are a helpful assistant. Reasoning: {request.reasoning}"
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": request.message}
        ]
        
        # Call Ollama via OpenAI-compatible API
        response = ollama_client.chat.completions.create(
            model=selected_model,
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=request.stream
        )
        
        if request.stream:
            # Handle streaming response
            def generate():
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        yield f"data: {chunk.choices[0].delta.content}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(generate(), media_type="text/plain")
        else:
            # Handle regular response
            content = response.choices[0].message.content
            usage_info = {
                "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                "total_tokens": getattr(response.usage, 'total_tokens', 0)
            } if hasattr(response, 'usage') and response.usage else {}
            
            return ChatResponse(
                response=content,
                model=selected_model,
                usage=usage_info
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response with model {selected_model}: {str(e)}")

# Specific model endpoints
@app.post("/chat/{model_name}", response_model=ChatResponse)
async def chat_with_specific_model(model_name: str, request: ChatRequest):
    """Chat with a specific model by name"""
    if ollama_client is None:
        raise HTTPException(status_code=503, detail="Ollama client not initialized")
    
    # Validate model exists
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400, 
            detail=f"Model {model_name} not available. Available models: {list(AVAILABLE_MODELS.keys())}"
        )
    
    # Override the request model with the path parameter
    request.model = model_name
    return await chat(request)

# Convenience endpoints for each model
@app.post("/chat/qwen3", response_model=ChatResponse)
async def chat_qwen3(request: ChatRequest):
    """Chat with Qwen 3.0 30B model"""
    request.model = "qwen3:30b"
    return await chat(request)

@app.post("/chat/devstral", response_model=ChatResponse) 
async def chat_devstral(request: ChatRequest):
    """Chat with DevStral model"""
    request.model = "devstral"
    return await chat(request)

@app.post("/chat/gemma3", response_model=ChatResponse)
async def chat_gemma3(request: ChatRequest):
    """Chat with Gemma 3 model"""
    request.model = "gemma3"
    return await chat(request)

@app.post("/chat/gpt-oss", response_model=ChatResponse)
async def chat_gpt_oss(request: ChatRequest):
    """Chat with GPT-OSS 20B model"""
    request.model = "gpt-oss:20b"
    return await chat(request)

# OpenAI-compatible endpoint (direct proxy)
@app.post("/v1/chat/completions")
async def openai_compatible_chat(request: dict):
    """Direct OpenAI-compatible endpoint"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ollama_base_url}/v1/chat/completions",
                json=request,
                timeout=300.0
            )
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in OpenAI-compatible endpoint: {str(e)}")

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """Create embeddings using Ollama"""
    if ollama_client is None:
        raise HTTPException(status_code=503, detail="Ollama client not initialized")
    
    try:
        # Handle both string and list input
        texts = [request.input] if isinstance(request.input, str) else request.input
        
        # Generate embeddings for each text
        embedding_data = []
        total_tokens = 0
        
        for i, text in enumerate(texts):
            # Use Ollama's embedding endpoint directly
            async with httpx.AsyncClient(timeout=300.0) as client:
                # Use the requested model or default for embeddings
                embedding_model = request.model if request.model and request.model in AVAILABLE_MODELS else default_model
                ollama_request = {
                    "model": embedding_model,
                    "prompt": text
                }
                
                response = await client.post(
                    f"{ollama_base_url}/api/embeddings",
                    json=ollama_request
                )
                
                if response.status_code == 200:
                    result = response.json()
                    embedding = result.get("embedding", [])
                    
                    # If no embedding returned, create a dummy one (1536 dimensions)
                    if not embedding:
                        embedding = [0.0] * 1536
                    
                    embedding_data.append(EmbeddingData(
                        index=i,
                        embedding=embedding
                    ))
                    
                    # Estimate tokens (rough approximation)
                    total_tokens += len(text.split())
                else:
                    # Fallback: create dummy embedding if Ollama doesn't support embeddings
                    print(f"⚠️ Ollama embeddings not available, using dummy embedding for: {text[:50]}...")
                    embedding_data.append(EmbeddingData(
                        index=i,
                        embedding=[0.0] * 1536  # Standard OpenAI embedding size
                    ))
                    total_tokens += len(text.split())
        
        usage_info = {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens
        }
        
        return EmbeddingResponse(
            data=embedding_data,
            model=request.model or default_model,
            usage=usage_info
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting Multi-Model API Gateway...")
    print("🌐 Local API URL: http://localhost:8000")
    print("🌐 Web UI URL: http://localhost:3000")
    uvicorn.run(app, host="0.0.0.0", port=8000)