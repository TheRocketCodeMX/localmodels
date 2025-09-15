import os
import asyncio
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import uvicorn
import threading
import time
import litellm
from litellm import completion, acompletion
from typing import Optional, List, Dict, Any

app = FastAPI(title="Multi-Model API with LiteLLM", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
default_model = os.getenv("DEFAULT_MODEL", "gpt-oss:20b")

# Available models configuration
AVAILABLE_MODELS = {
    "gpt-oss:20b": {
        "name": "gpt-oss:20b", 
        "description": "GPT-OSS 20B model",
        "provider": "ollama",
        "litellm_name": "ollama/gpt-oss:20b"
    },
    "qwen3:30b": {
        "name": "qwen3:30b", 
        "description": "Qwen 3.0 30B model",
        "provider": "ollama", 
        "litellm_name": "ollama/qwen3:30b"
    },
    "devstral": {
        "name": "devstral", 
        "description": "DevStral model",
        "provider": "ollama",
        "litellm_name": "ollama/devstral"
    },
    "gemma3": {
        "name": "gemma3", 
        "description": "Gemma 3 model", 
        "provider": "ollama",
        "litellm_name": "ollama/gemma3"
    }
}

# Configure LiteLLM
litellm.set_verbose = True  # Enable debug
litellm.drop_params = True
# Turn on debug
import litellm
litellm._turn_on_debug()

# Ollama client for direct access
ollama_client = None

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

class OpenAIChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

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
        async with httpx.AsyncClient(timeout=1800.0) as client:
            response = await client.get(f"{ollama_base_url}/api/tags")
            existing_models = response.json().get("models", [])
            existing_model_names = [model["name"] for model in existing_models]
            
            for model_key, model_info in AVAILABLE_MODELS.items():
                model_name = model_info["name"]
                
                model_exists = any(model_name in existing for existing in existing_model_names)
                
                if not model_exists:
                    print(f"📥 Downloading model {model_name}... This may take 10-30 minutes.")
                    print(f"💡 Large models may require significant disk space and RAM")
                    
                    pull_data = {"name": model_name, "stream": True}
                    
                    async with client.stream("POST", f"{ollama_base_url}/api/pull", json=pull_data) as pull_response:
                        if pull_response.status_code == 200:
                            async for line in pull_response.aiter_lines():
                                if line:
                                    try:
                                        data = eval(line)
                                        if "status" in data:
                                            print(f"📡 {model_name}: {data['status']}")
                                        if data.get("status") == "success":
                                            print(f"✅ Model {model_name} downloaded successfully!")
                                            break
                                    except:
                                        continue
                        else:
                            print(f"⚠️ Failed to download model {model_name}. Status: {pull_response.status_code}")
                else:
                    print(f"✅ Model {model_name} already available")
            
            return True
                
    except Exception as e:
        print(f"❌ Error ensuring model availability: {e}")
        return True

def setup_ollama_client():
    """Setup OpenAI client pointing to Ollama"""
    global ollama_client
    
    ollama_client = OpenAI(
        base_url=f"{ollama_base_url}/v1",
        api_key="ollama"
    )
    print("✅ Ollama client configured")

def setup_litellm():
    """Configure LiteLLM for Ollama"""
    # Set the Ollama base URL for LiteLLM
    os.environ["OLLAMA_API_BASE"] = ollama_base_url
    print("✅ LiteLLM configured for Ollama")

@app.on_event("startup")
async def startup_event():
    print("🚀 Starting Multi-Model API Gateway with LiteLLM...")
    
    # Wait for Ollama service
    if not await wait_for_ollama():
        raise RuntimeError("Could not connect to Ollama service")
    
    # Setup clients
    setup_ollama_client()
    setup_litellm()
    
    # Ensure models are available
    await ensure_models_available()
    
    print("✅ Startup complete!")
    print("🌐 Local API URL: http://localhost:8000")
    print("🌐 LiteLLM Proxy: http://localhost:4000") 
    print("🌐 Web UI URL: http://localhost:3000")

@app.get("/")
async def root():
    return {
        "message": "Multi-Model API with LiteLLM is running",
        "default_model": default_model,
        "available_models": AVAILABLE_MODELS,
        "ollama_url": ollama_base_url,
        "endpoints": {
            "chat": "/chat",
            "chat_with_model": "/chat/{model_name}",
            "chat_litellm": "/v1/chat/completions",
            "embeddings": "/v1/embeddings", 
            "health": "/health",
            "models": "/models"
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
            "available_models": list(AVAILABLE_MODELS.keys()),
            "litellm_enabled": True
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
    """Chat endpoint using LiteLLM"""
    selected_model = request.model if request.model and request.model in AVAILABLE_MODELS else default_model
    
    try:
        # Get the LiteLLM model name
        litellm_model = AVAILABLE_MODELS[selected_model]["litellm_name"]
        
        # Format message with reasoning level
        messages = [
            {"role": "system", "content": f"You are a helpful assistant. Reasoning: {request.reasoning}"},
            {"role": "user", "content": request.message}
        ]
        
        if request.stream:
            # Handle streaming with LiteLLM
            async def generate():
                try:
                    response = await acompletion(
                        model=litellm_model,
                        messages=messages,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        stream=True,
                        api_base=ollama_base_url
                    )
                    
                    async for chunk in response:
                        if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                            yield f"data: {chunk.choices[0].delta.content}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    yield f"data: Error: {str(e)}\n\n"
            
            return StreamingResponse(generate(), media_type="text/plain")
        else:
            # Handle regular response with LiteLLM
            try:
                response = await acompletion(
                    model=litellm_model,
                    messages=messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    api_base=ollama_base_url,
                    timeout=60
                )
            except Exception as litellm_error:
                print(f"LiteLLM Error: {litellm_error}")
                # Fallback: try direct Ollama call
                try:
                    async with httpx.AsyncClient(timeout=60.0) as client:
                        ollama_request = {
                            "model": selected_model,
                            "prompt": request.message,
                            "options": {
                                "num_predict": request.max_tokens,
                                "temperature": request.temperature
                            },
                            "stream": False
                        }
                        
                        direct_response = await client.post(
                            f"{ollama_base_url}/api/generate",
                            json=ollama_request
                        )
                        
                        if direct_response.status_code == 200:
                            result = direct_response.json()
                            return ChatResponse(
                                response=result.get("response", ""),
                                model=selected_model,
                                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                            )
                        else:
                            raise HTTPException(status_code=500, detail="Both LiteLLM and direct Ollama failed")
                            
                except Exception as fallback_error:
                    raise HTTPException(status_code=500, detail=f"All methods failed: LiteLLM: {litellm_error}, Direct: {fallback_error}")
            
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
        raise HTTPException(status_code=500, detail=f"Error generating response with LiteLLM model {selected_model}: {str(e)}")

# OpenAI-compatible endpoint using LiteLLM
@app.post("/v1/chat/completions")
async def litellm_chat_completions(request: OpenAIChatRequest):
    """OpenAI-compatible endpoint using LiteLLM"""
    try:
        # Validate model
        if request.model not in AVAILABLE_MODELS:
            raise HTTPException(
                status_code=400, 
                detail=f"Model {request.model} not available. Available models: {list(AVAILABLE_MODELS.keys())}"
            )
        
        # Get the LiteLLM model name
        litellm_model = AVAILABLE_MODELS[request.model]["litellm_name"]
        
        if request.stream:
            # Handle streaming
            async def generate():
                try:
                    response = await acompletion(
                        model=litellm_model,
                        messages=request.messages,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        stream=True,
                        api_base=ollama_base_url
                    )
                    
                    async for chunk in response:
                        yield f"data: {chunk.model_dump_json()}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    yield f"data: {{'error': '{str(e)}'}}\n\n"
            
            return StreamingResponse(generate(), media_type="text/plain")
        else:
            # Handle regular response
            response = await acompletion(
                model=litellm_model,
                messages=request.messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                api_base=ollama_base_url
            )
            
            return response.model_dump()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in LiteLLM endpoint: {str(e)}")

# Specific model endpoints
@app.post("/chat/{model_name}", response_model=ChatResponse)
async def chat_with_specific_model(model_name: str, request: ChatRequest):
    """Chat with a specific model by name"""
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400, 
            detail=f"Model {model_name} not available. Available models: {list(AVAILABLE_MODELS.keys())}"
        )
    
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

# Direct Ollama endpoint for testing (without LiteLLM)
@app.post("/chat/direct", response_model=ChatResponse)
async def chat_direct_ollama(request: ChatRequest):
    """Direct chat with Ollama (bypassing LiteLLM)"""
    selected_model = request.model if request.model and request.model in AVAILABLE_MODELS else default_model
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            ollama_request = {
                "model": selected_model,
                "prompt": f"Reasoning: {request.reasoning}\nUser: {request.message}\nAssistant:",
                "options": {
                    "num_predict": request.max_tokens,
                    "temperature": request.temperature
                },
                "stream": False
            }
            
            response = await client.post(
                f"{ollama_base_url}/api/generate",
                json=ollama_request
            )
            
            if response.status_code == 200:
                result = response.json()
                return ChatResponse(
                    response=result.get("response", ""),
                    model=selected_model,
                    usage={
                        "prompt_tokens": result.get("prompt_eval_count", 0),
                        "completion_tokens": result.get("eval_count", 0),
                        "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                    }
                )
            else:
                raise HTTPException(status_code=500, detail=f"Ollama API error: {response.status_code}")
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error with direct Ollama call: {str(e)}")

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """Create embeddings using Ollama"""
    if ollama_client is None:
        raise HTTPException(status_code=503, detail="Ollama client not initialized")
    
    try:
        texts = [request.input] if isinstance(request.input, str) else request.input
        embedding_data = []
        total_tokens = 0
        
        for i, text in enumerate(texts):
            async with httpx.AsyncClient(timeout=300.0) as client:
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
                    
                    if not embedding:
                        embedding = [0.0] * 1536
                    
                    embedding_data.append(EmbeddingData(
                        index=i,
                        embedding=embedding
                    ))
                    
                    total_tokens += len(text.split())
                else:
                    print(f"⚠️ Ollama embeddings not available, using dummy embedding for: {text[:50]}...")
                    embedding_data.append(EmbeddingData(
                        index=i,
                        embedding=[0.0] * 1536
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
    print("🚀 Starting Multi-Model API Gateway with LiteLLM...")
    print("🌐 Local API URL: http://localhost:8000")
    print("🌐 Web UI URL: http://localhost:3000")
    uvicorn.run(app, host="0.0.0.0", port=8000)