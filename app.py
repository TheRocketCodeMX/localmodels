import os
import asyncio
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
import uvicorn
from pyngrok import ngrok
import threading
import time

app = FastAPI(title="GPT-OSS API via Ollama", version="1.0.0")

# Ollama client
ollama_client = None
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
gpt_oss_model = os.getenv("GPT_OSS_MODEL", "gpt-oss:20b")

class ChatRequest(BaseModel):
    message: str
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

async def ensure_model_available():
    """Download GPT-OSS model if not available"""
    try:
        # Use longer timeout for model download (30 minutes)
        async with httpx.AsyncClient(timeout=1800.0) as client:
            # Check if model is already available
            response = await client.get(f"{ollama_base_url}/api/tags")
            models = response.json().get("models", [])
            
            model_exists = any(gpt_oss_model in model["name"] for model in models)
            
            if not model_exists:
                print(f"📥 Downloading model {gpt_oss_model}... This may take 10-20 minutes.")
                print(f"💡 Model size: ~10GB for gpt-oss:20b or ~60GB for gpt-oss:120b")
                
                # Pull the model with streaming to show progress
                pull_data = {"name": gpt_oss_model, "stream": True}
                
                async with client.stream("POST", f"{ollama_base_url}/api/pull", json=pull_data) as response:
                    if response.status_code == 200:
                        async for line in response.aiter_lines():
                            if line:
                                try:
                                    data = eval(line)  # Parse JSON line
                                    if "status" in data:
                                        print(f"📡 {data['status']}")
                                    if data.get("status") == "success":
                                        print(f"✅ Model {gpt_oss_model} downloaded successfully!")
                                        return True
                                except:
                                    continue
                    else:
                        print(f"❌ Failed to download model. Status: {response.status_code}")
                        print(f"Error: {response.text}")
                        return False
            else:
                print(f"✅ Model {gpt_oss_model} already available")
                return True
                
    except Exception as e:
        print(f"❌ Error ensuring model availability: {e}")
        # Don't fail completely, allow manual download
        print(f"💡 You can manually download with: docker exec -it ollmalocally-ollama-1 ollama pull {gpt_oss_model}")
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
    
    # Ensure model is available
    await ensure_model_available()
    
    print("✅ Startup complete!")

@app.get("/")
async def root():
    return {
        "message": "GPT-OSS API via Ollama is running",
        "model": gpt_oss_model,
        "ollama_url": ollama_base_url,
        "endpoints": {
            "chat": "/chat",
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
            "model": gpt_oss_model
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
    
    try:
        # Format message with reasoning level
        system_content = f"You are a helpful assistant. Reasoning: {request.reasoning}"
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": request.message}
        ]
        
        # Call Ollama via OpenAI-compatible API
        response = ollama_client.chat.completions.create(
            model=gpt_oss_model,
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
                model=gpt_oss_model,
                usage=usage_info
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

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
                ollama_request = {
                    "model": gpt_oss_model,
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
            model=request.model or gpt_oss_model,
            usage=usage_info
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

def start_ngrok():
    """Start ngrok tunnel"""
    ngrok_token = os.getenv("NGROK_AUTH_TOKEN")
    if ngrok_token:
        try:
            ngrok.set_auth_token(ngrok_token)
            tunnel = ngrok.connect(8000)
            print(f"🌐 Public URL: {tunnel.public_url}")
            print(f"🌐 Chat endpoint: {tunnel.public_url}/chat")
            print(f"🌐 Embeddings endpoint: {tunnel.public_url}/v1/embeddings")
            print(f"🌐 OpenAI compatible: {tunnel.public_url}/v1/chat/completions")
        except Exception as e:
            print(f"⚠️  Ngrok setup failed: {e}")
    else:
        print("⚠️  NGROK_AUTH_TOKEN not set. Running without ngrok.")

if __name__ == "__main__":
    # Start ngrok in separate thread
    ngrok_thread = threading.Thread(target=start_ngrok)
    ngrok_thread.daemon = True
    ngrok_thread.start()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)