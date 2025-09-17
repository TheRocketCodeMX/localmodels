import os
import asyncio
import httpx
import time
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import uvicorn
import litellm
from litellm import completion, acompletion
from typing import Optional, List, Dict, Any

app = FastAPI(title="Multi-Model API with LiteLLM", version="6.0.0")

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
litellm.max_budget = 100
litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]
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
    os.environ["OLLAMA_API_KEY"] = "ollama" # Added OLLAMA_API_KEY

    # Configure global LiteLLM timeout and retries
    litellm_timeout = int(os.getenv("LITELLM_TIMEOUT", 600)) # Default to 600 seconds (10 minutes)
    litellm_retries = int(os.getenv("LITELLM_RETRIES", 3)) # Default to 3 retries

    litellm.set_timeout(litellm_timeout)
    litellm.max_retries = litellm_retries

    print(f"✅ LiteLLM configured for Ollama with timeout={litellm_timeout}s and retries={litellm_retries}")

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
                    timeout=180 # Increased timeout
                )
            except Exception as litellm_error:
                print(f"LiteLLM Error: {litellm_error}")
                # Fallback: try direct Ollama call
                try:
                    async with httpx.AsyncClient(timeout=180.0) as client: # Increased timeout
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
                api_base=ollama_base_url,
                timeout=180 # Increased timeout
            )
            
            return response.model_dump()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in LiteLLM endpoint: {str(e)}")

# Direct Ollama endpoint for testing (without LiteLLM)
@app.post("/chat/direct", response_model=ChatResponse)
async def chat_direct_ollama(request: ChatRequest):
    """Direct chat with Ollama (bypassing LiteLLM)"""
    selected_model = request.model if request.model and request.model in AVAILABLE_MODELS else default_model
    
    try:
        async with httpx.AsyncClient(timeout=180.0) as client: # Increased timeout
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

@app.post("/debug/litellm/{model_name}", include_in_schema=False)
async def debug_litellm_model(model_name: str, request: ChatRequest) -> Response:
    """
    Enhanced debug endpoint to test a specific model with LiteLLM directly,
    providing curated, relevant performance and hardware context.
    """
    if model_name not in AVAILABLE_MODELS:
        return JSONResponse(
            content={"status": "error", "error_details": {"error_message": f"Model '{model_name}' not found."}},
            status_code=404
        )

    # Step 1: Get and curate model and runtime hardware details from Ollama
    hardware_info = {}
    warning_message = None

    async def _normalize_name(n: str) -> str:
        if not n:
            return ""
        n = n.lower()
        # strip common prefixes (e.g., 'ollama/', registries)
        for prefix in ["ollama/", "registry.ollama.ai/library/", "library/"]:
            if n.startswith(prefix):
                n = n[len(prefix):]
        return n

    async def gather_runtime_info() -> tuple[dict, Optional[str]]:
        try:
            def _bytes_to_gb(x: Optional[float | int]) -> Optional[float]:
                try:
                    if x is None:
                        return None
                    return round(float(x) / (1024 ** 3), 2)
                except Exception:
                    return None

            async with httpx.AsyncClient() as client:
                ollama_model_name = AVAILABLE_MODELS[model_name]["name"]
                norm_target = (await _normalize_name(ollama_model_name))

                def _norm_local(n: Optional[str]) -> str:
                    if not n:
                        return ""
                    s = n.lower()
                    for prefix in ["ollama/", "registry.ollama.ai/library/", "library/"]:
                        if s.startswith(prefix):
                            s = s[len(prefix):]
                    return s

                # 1) Static model metadata (/api/show)
                show_resp = await client.post(
                    f"{ollama_base_url}/api/show",
                    json={"name": ollama_model_name},
                    timeout=30,
                )
                raw_details = show_resp.json() if show_resp.status_code == 200 else {}
                details_section = raw_details.get("details", {})
                library = (raw_details.get("library") or "").lower()

                # 2) Runtime process info (/api/ps)
                ps_resp = await client.get(f"{ollama_base_url}/api/ps", timeout=10)
                ps_json = ps_resp.json() if ps_resp.status_code == 200 else {}
                ps_models = ps_json.get("models", []) if isinstance(ps_json, dict) else []

                # Extract GPU device names directly from /api/ps top-level (preferred over heuristics)
                gpu_devices: list[dict] = []
                def _extract_devices(obj):
                    try:
                        if isinstance(obj, dict):
                            for k, v in obj.items():
                                lk = str(k).lower()
                                if lk in ["devices", "gpus", "gpu_devices", "hardware"]:
                                    # handle common layouts: {hardware:{gpus:[...]}} or direct {gpus:[...]}
                                    if isinstance(v, dict):
                                        # recurse into hardware dict
                                        _extract_devices(v)
                                    elif isinstance(v, list):
                                        for d in v:
                                            if isinstance(d, dict):
                                                name = d.get("name") or d.get("device_name") or d.get("model")
                                                lib = (d.get("library") or d.get("backend") or d.get("llm_library") or "").lower()
                                                if isinstance(name, str) and name.strip():
                                                    gpu_devices.append({"name": name.strip(), "library": lib})
                                            elif isinstance(d, str) and d.strip():
                                                # list of strings with GPU names
                                                gpu_devices.append({"name": d.strip(), "library": ""})
                                    else:
                                        # continue walking for unexpected structures
                                        _extract_devices(v)
                                else:
                                    _extract_devices(v)
                        elif isinstance(obj, list):
                            for it in obj:
                                _extract_devices(it)
                    except Exception:
                        pass
                _extract_devices(ps_json)

                ps_entry = None
                for m in ps_models:
                    # some installs report "model" or "name"
                    m_name = m.get("name") or m.get("model")
                    if _norm_local(m_name) == norm_target:
                        ps_entry = m
                        break
                # Heuristic: if only one model loaded, use it
                if not ps_entry and len(ps_models) == 1:
                    ps_entry = ps_models[0]

                # 3) Version info (optional)
                version = None
                try:
                    v_resp = await client.get(f"{ollama_base_url}/api/version", timeout=5)
                    if v_resp.status_code == 200:
                        v = v_resp.json()
                        version = v.get("version") if isinstance(v, dict) else None
                except Exception:
                    pass

                # Heuristics to detect runtime mode
                total_layers = details_section.get("total_layers") or 0
                offloaded_layers = 0
                gpu_layers = None

                # Common fields we may see in /api/ps
                size_total = None
                size_vram = None
                gpu_name: Optional[str] = None

                if ps_entry:
                    size_total = ps_entry.get("size") or ps_entry.get("size_total")
                    size_vram = ps_entry.get("size_vram") or ps_entry.get("vram_size")
                    gpu_layers = ps_entry.get("gpu_layers") or ps_entry.get("layers_on_gpu")
                    # try to get library from runtime entry if available
                    lib_candidate = ps_entry.get("library") or ps_entry.get("llm_library") or ps_entry.get("backend")
                    if isinstance(lib_candidate, str) and lib_candidate.strip():
                        library = lib_candidate.lower()
                    # prefer explicit gpu fields; avoid generic 'name' which is model name
                    for key in ["gpu", "gpu_name", "device", "device_name", "adapter", "adapter_name"]:
                        val = ps_entry.get(key)
                        if isinstance(val, dict):
                            val = val.get("name") or val.get("model") or val.get("device_name")
                        if isinstance(val, str) and val.strip():
                            # ignore if equal to model name
                            if _norm_local(val) != norm_target:
                                gpu_name = val
                                break

                # If not found inside the model entry, scan ps_json more broadly
                def _walk(obj):
                    nonlocal gpu_name, library
                    try:
                        if isinstance(obj, dict):
                            # consider only fields that look gpu/library-related
                            for k, v in obj.items():
                                lk = str(k).lower()
                                # pick up library hints
                                if lk in ["library", "llm_library", "backend"] and isinstance(v, str):
                                    lv = v.lower().strip()
                                    if lv in [
                                        "cuda", "rocm", "hip", "gpu",
                                        "intel", "oneapi", "level-zero", "level_zero", "levelzero",
                                        "metal", "mps", "apple",
                                        "opencl", "vulkan"
                                    ]:
                                        library = lv
                                # pick up possible gpu model strings
                                if lk in ["gpu", "gpu_name", "device", "device_name", "adapter", "adapter_name", "name", "model"]:
                                    s = None
                                    if isinstance(v, dict):
                                        s = v.get("name") or v.get("model") or v.get("device_name") or v.get("adapter_name")
                                    elif isinstance(v, str):
                                        s = v
                                    if isinstance(s, str) and s.strip():
                                        ns = s.strip()
                                        # skip if it matches model name
                                        if _norm_local(ns) != norm_target:
                                            # prefer strings that look like a GPU brand
                                            lns = ns.lower()
                                            if any(t in lns for t in [
                                                "nvidia", "geforce", "rtx", "gtx",
                                                "cuda", "rocm", "hip", "amd", "radeon",
                                                "intel", "arc", "iris", "uhd", "xe",
                                                "apple", "metal", "mps", "m1", "m2", "m3", "m4"
                                            ]):
                                                if not gpu_name:
                                                    gpu_name = ns
                                # Also consider any arbitrary string value for vendor/library hints
                                if isinstance(v, str):
                                    sv = v.strip()
                                    if sv and _norm_local(sv) != norm_target:
                                        lsv = sv.lower()
                                        if any(tok in lsv for tok in ["nvidia", "geforce", "rtx", "gtx"]):
                                            if not gpu_name:
                                                gpu_name = sv
                                            if not library:
                                                library = "cuda"
                                        elif any(tok in lsv for tok in ["amd", "radeon", "rocm", "hip"]):
                                            if not gpu_name:
                                                gpu_name = sv
                                            if not library:
                                                library = "rocm"
                                        elif any(tok in lsv for tok in ["intel", "oneapi", "level-zero", "level_zero", "levelzero", "xe", "arc"]):
                                            if not gpu_name:
                                                gpu_name = sv
                                            if not library:
                                                library = "intel"
                                        elif any(tok in lsv for tok in ["apple", "metal", "mps"]):
                                            if not gpu_name:
                                                gpu_name = sv
                                            if not library:
                                                library = "metal"
                                # continue walking
                                _walk(v)
                        elif isinstance(obj, list):
                            for it in obj:
                                _walk(it)
                    except Exception:
                        pass

                if not gpu_name and isinstance(ps_json, (dict, list)):
                    _walk(ps_json)

                # Prefer explicit device list from /api/ps if available
                if not gpu_name and gpu_devices:
                    gpu_name = gpu_devices[0].get("name")
                    if not library:
                        lib0 = gpu_devices[0].get("library")
                        if isinstance(lib0, str) and lib0.strip():
                            library = lib0

                # Fallbacks
                if not gpu_name:
                    gpu_name = details_section.get("gpu_model")
                if not library:
                    # infer library from gpu_name or other hints; avoid assuming CUDA by default
                    if gpu_name:
                        l = gpu_name.lower()
                        if ("nvidia" in l or "geforce" in l or "rtx" in l or "gtx" in l or "cuda" in l):
                            library = "cuda"
                        elif ("amd" in l or "radeon" in l or "rocm" in l or "hip" in l or "gfx" in l):
                            library = "rocm"
                        elif ("intel" in l or "arc" in l or "iris" in l or "uhd" in l or "xe" in l or "oneapi" in l or "level" in l):
                            library = "intel"
                        elif ("apple" in l or "metal" in l or "mps" in l or any(t in l for t in ["m1","m2","m3","m4"])):
                            library = "metal"
                        else:
                            library = "gpu"
                    elif size_vram and float(size_vram) > 0:
                        # VRAM in use but no vendor signals: keep generic 'gpu'
                        library = "gpu"
                    else:
                        library = "cpu"

                # infer offloaded_layers from gpu_layers
                if isinstance(gpu_layers, int):
                    offloaded_layers = gpu_layers
                else:
                    offloaded_layers = details_section.get("offloaded_layers") or 0

                # Mode detection
                detected_mode = "CPU"
                if size_total and size_vram is not None:
                    try:
                        ratio = float(size_vram) / float(size_total) if float(size_total) > 0 else 0.0
                        if ratio == 0:
                            detected_mode = "CPU"
                        elif 0 < ratio < 0.98:
                            detected_mode = "Hybrid (CPU+GPU)"
                        else:
                            detected_mode = "GPU"
                    except Exception:
                        pass
                # If still ambiguous, infer from libraries and layers
                if detected_mode == "CPU":
                    if (("cuda" in library or "rocm" in library or library == "gpu") or gpu_name):
                        if total_layers and 0 < offloaded_layers < total_layers:
                            detected_mode = "Hybrid (CPU+GPU)"
                        elif total_layers and offloaded_layers >= total_layers > 0:
                            detected_mode = "GPU"
                        elif isinstance(gpu_layers, int) and gpu_layers > 0:
                            detected_mode = "Hybrid (CPU+GPU)"

                # Build warnings
                warn = None
                if detected_mode == "CPU" and (("cuda" in library or "rocm" in library or library == "gpu") or gpu_name):
                    warn = "GPU library detected but no VRAM usage reported; model appears to be running on CPU."

                # Compute human-readable memory fields
                size_total_gb = _bytes_to_gb(size_total)
                size_vram_gb = _bytes_to_gb(size_vram)
                vram_used_bytes = size_vram
                vram_used_gb = size_vram_gb
                cpu_ram_used_bytes = None
                cpu_ram_used_gb = None
                try:
                    if size_total is not None and size_vram is not None:
                        cpu_ram_used_bytes = max(0, int(float(size_total) - float(size_vram)))
                        cpu_ram_used_gb = _bytes_to_gb(cpu_ram_used_bytes)
                except Exception:
                    pass

                # Choose a final GPU model string
                fallback_gpu = None

                # System-level GPU detection as a last resort when ps/json didn't expose it
                def _probe_system_gpu_name(lib_hint: str | None) -> Optional[str]:
                    try:
                        import shutil, subprocess, json as _json
                        # Prefer vendor-specific tools based on hint
                        lh = (lib_hint or "").lower()
                        # Try NVIDIA first (if CUDA hinted or no hint at all)
                        if (not lh or "cuda" in lh or "gpu" == lh) and shutil.which("nvidia-smi"):
                            try:
                                out = subprocess.check_output([
                                    "nvidia-smi", "--query-gpu=name", "--format=csv,noheader"
                                ], stderr=subprocess.DEVNULL).decode().strip()
                                if out:
                                    # if multiple lines/devices, pick first but keep full list if needed later
                                    first = out.splitlines()[0].strip()
                                    if first:
                                        return first
                            except Exception:
                                pass
                        # Try AMD ROCm
                        if (not lh or "rocm" in lh or "hip" in lh or "gpu" == lh) and shutil.which("rocm-smi"):
                            try:
                                # rocm-smi --json is supported on recent versions
                                out = subprocess.check_output(["rocm-smi", "--json"], stderr=subprocess.DEVNULL).decode()
                                try:
                                    j = _json.loads(out)
                                    # Look for product or name fields
                                    # Newer rocm-smi places devices under key like "card" or list entries
                                    def _find_amd_name(o):
                                        if isinstance(o, dict):
                                            for k,v in o.items():
                                                kl = str(k).lower()
                                                if kl in ("card", "device", "gpu", "cards", "devices"):
                                                    n = _find_amd_name(v)
                                                    if n:
                                                        return n
                                                if kl in ("name", "product", "product_name", "marketing_name") and isinstance(v, str) and v.strip():
                                                    return v.strip()
                                            return None
                                        if isinstance(o, list):
                                            for it in o:
                                                n = _find_amd_name(it)
                                                if n:
                                                    return n
                                        return None
                                    name = _find_amd_name(j)
                                    if name:
                                        return name
                                except Exception:
                                    pass
                                # Fallback textual parse
                                out2 = subprocess.check_output(["rocm-smi", "--showproductname"], stderr=subprocess.DEVNULL).decode()
                                for line in out2.splitlines():
                                    ls = line.strip()
                                    if ls and ("card" in ls.lower() or "product" in ls.lower()):
                                        # extract after ':'
                                        if ":" in ls:
                                            cand = ls.split(":",1)[1].strip()
                                            if cand:
                                                return cand
                            except Exception:
                                pass
                        # Try Intel oneAPI/Level Zero hints
                        if (not lh or "intel" in lh or "oneapi" in lh or "level" in lh or "gpu" == lh):
                            # intel_gpu_top -J gives JSON if available
                            if shutil.which("intel_gpu_top"):
                                try:
                                    out = subprocess.check_output(["intel_gpu_top", "-J"], stderr=subprocess.DEVNULL, timeout=2).decode()
                                    j = _json.loads(out)
                                    # try to extract device name
                                    name = None
                                    if isinstance(j, dict):
                                        name = j.get("SYS", {}).get("device_name") or j.get("device_name")
                                    if isinstance(name, str) and name.strip():
                                        return name.strip()
                                except Exception:
                                    pass
                            # As a last resort, use lspci if present
                            if shutil.which("lspci"):
                                try:
                                    out = subprocess.check_output(["sh", "-lc", "lspci -nn | grep -i 'vga\|3d'"], stderr=subprocess.DEVNULL).decode()
                                    for line in out.splitlines():
                                        ls = line.strip()
                                        if "intel" in ls.lower():
                                            # take text after controller label
                                            parts = ls.split(":", 2)
                                            cand = parts[-1].strip() if parts else ls
                                            if cand:
                                                return cand
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    return None

                if not gpu_name:
                    # Try probing system tools based on current library hint
                    sys_gpu = _probe_system_gpu_name(library)
                    if isinstance(sys_gpu, str) and sys_gpu.strip():
                        gpu_name = sys_gpu.strip()

                if not gpu_name:
                    if isinstance(library, str):
                        ll = library.lower()
                        if "cuda" in ll:
                            fallback_gpu = "NVIDIA (CUDA)"
                        elif ("rocm" in ll or "hip" in ll):
                            fallback_gpu = "AMD (ROCm/HIP)"
                        elif ("intel" in ll or "oneapi" in ll or "level" in ll):
                            fallback_gpu = "Intel (oneAPI)"
                        elif ("metal" in ll or "mps" in ll or "apple" in ll):
                            fallback_gpu = "Apple (Metal)"
                        elif ("opencl" in ll):
                            fallback_gpu = "GPU (OpenCL)"
                        elif ("vulkan" in ll):
                            fallback_gpu = "GPU (Vulkan)"
                    if not fallback_gpu and size_vram and float(size_vram) > 0:
                        # Generic, vendor-agnostic fallback when VRAM is in use but no brand/library detected
                        fallback_gpu = "GPU (unknown vendor)"
                info = {
                    "detected_mode": detected_mode,
                    "gpu_model": gpu_name or fallback_gpu or "N/A",
                    "gpu_layers_offloaded": f"{offloaded_layers}/{total_layers}" if total_layers else (offloaded_layers or 0),
                    "size_total_bytes": size_total,
                    "size_vram_bytes": size_vram,
                    "size_total_gb": size_total_gb,
                    "size_vram_gb": size_vram_gb,
                    "vram_used_bytes": vram_used_bytes,
                    "vram_used_gb": vram_used_gb,
                    "cpu_ram_used_bytes": cpu_ram_used_bytes,
                    "cpu_ram_used_gb": cpu_ram_used_gb,
                    "parameter_size": details_section.get("parameter_size"),
                    "quantization_level": details_section.get("quantization_level"),
                    "ollama_library": library or "cpu",
                    "ollama_version": version,
                    "source": {
                        "show": show_resp.status_code if 'show_resp' in locals() else None,
                        "ps": ps_resp.status_code if 'ps_resp' in locals() else None,
                        "version": 200 if version else None
                    }
                }
                return info, warn
        except Exception as ex:
            return {"error": f"Could not collect hardware/runtime details: {str(ex)}"}, None

    # First pass (may be before model loads)
    hardware_info, warning_message = await gather_runtime_info()

    # Step 2: Execute the LiteLLM completion request
    litellm_model_name = AVAILABLE_MODELS[model_name]["litellm_name"]
    messages = [
        {"role": "system", "content": f"You are a helpful assistant. Reasoning: {request.reasoning}"},
        {"role": "user", "content": request.message}
    ]

    request_payload = {
        "model": litellm_model_name,
        "messages": messages,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "api_base": ollama_base_url,
        "timeout": 180
    }

    start_time = time.time()
    try:
        print(f"--- Initiating Enhanced LiteLLM debug for model: {litellm_model_name} ---")
        response_obj = await acompletion(**request_payload)
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        print(f"--- LiteLLM debug successful (Latency: {latency_ms:.2f} ms) ---")

        # Step 3: Second pass to refresh runtime info post-execution
        post_info, post_warn = await gather_runtime_info()

        def _strength(info: dict) -> int:
            if not isinstance(info, dict):
                return 0
            strength = 0
            try:
                if info.get("size_vram_bytes"):
                    strength += 2
                gl = info.get("gpu_layers_offloaded")
                if isinstance(gl, str) and "/" in gl:
                    try:
                        num = int(gl.split("/")[0])
                        if num > 0:
                            strength += 1
                    except Exception:
                        pass
                elif isinstance(gl, int) and gl > 0:
                    strength += 1
                gm = info.get("gpu_model")
                if isinstance(gm, str) and gm not in ("N/A", ""):
                    strength += 1
            except Exception:
                pass
            return strength

        better_info = post_info if _strength(post_info) >= _strength(hardware_info) else hardware_info
        # Prefer any new warning if present
        warning_message = post_warn or warning_message

        # Step 4: Assemble the final success response
        final_content = {
            "status": "success",
            "debug_format_version": "6.0", # Final version
            "warning": warning_message,
            "latency_ms": round(latency_ms, 2),
            "hardware_info": better_info,
            "full_response_payload": response_obj.model_dump()
        }
        if warning_message is None:
            del final_content["warning"] # Keep the response clean

        return JSONResponse(content=final_content)

    except Exception as e:
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        print(f"--- LiteLLM debug FAILED (Latency: {latency_ms:.2f} ms) ---")

        # Step 3: Assemble the final error response
        error_content = {
            "status": "error",
            "debug_format_version": "6.0", # Final version
            "warning": warning_message,
            "latency_ms": round(latency_ms, 2),
            "hardware_info": hardware_info,
            "error_details": {
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
            "debug_info": {
                "model_used": litellm_model_name,
                "request_payload": request_payload
            }
        }
        if warning_message is None:
            del error_content["warning"] # Keep the response clean

        return JSONResponse(content=error_content, status_code=500)

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