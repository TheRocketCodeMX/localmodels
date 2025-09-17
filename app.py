import os
import asyncio
import httpx
import time
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI
import uvicorn
import litellm
from litellm import completion, acompletion
from typing import Optional, List, Dict, Any
import json
from logging_config import setup_logging # Import the setup_logging function
from contextlib import asynccontextmanager
# Gracefully import CircuitBreakerError from pybreaker; define fallback if not available
import importlib
try:
    CircuitBreakerError = importlib.import_module("pybreaker").CircuitBreakerError  # type: ignore[attr-defined]
except Exception:
    class CircuitBreakerError(Exception):
        pass

# Setup logging as early as possible
logger = setup_logging()

# Define FastAPI lifespan to replace deprecated on_event startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting Multi-Model API Gateway with LiteLLM...")
    # Wait for Ollama service
    if not await wait_for_ollama():
        raise RuntimeError("Could not connect to Ollama service")
    # Setup clients
    setup_ollama_client()
    setup_litellm()
    if use_litellm_proxy:
        setup_litellm_proxy_client()
    # Ensure models are available
    await ensure_models_available()
    logger.info("✅ Startup complete!")
    logger.info("🌐 Local API URL: http://localhost:8000")
    logger.info(f"🌐 LiteLLM Proxy: {litellm_proxy_url} (enabled={use_litellm_proxy})")
    logger.info("🌐 Web UI URL: http://localhost:3000")
    yield
    # Shutdown (no-op for now)

app = FastAPI(title="Multi-Model API with LiteLLM", version="6.0.0", lifespan=lifespan)

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
# LiteLLM Proxy configuration (standalone proxy)
use_litellm_proxy = os.getenv("USE_LITELLM_PROXY", "true").lower() in ("1", "true", "yes", "on")
litellm_proxy_url = os.getenv("LITELLM_PROXY_URL", "http://litellm-proxy:4000")
litellm_proxy_api_key = os.getenv("LITELLM_MASTER_KEY", "sk-1234")

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

# Simple async-friendly circuit breaker
class AsyncCircuitBreaker:
    def __init__(self, fail_max: int = 5, reset_timeout: float = 10.0, exclude: Optional[List[type]] = None):
        self.fail_max = fail_max
        self.reset_timeout = float(reset_timeout)
        self.exclude = tuple(exclude or [])
        self._fail_count = 0
        self._open_until = 0.0

    def _now(self) -> float:
        return time.time()

    def begin(self):
        # Raise if currently open
        if self.is_open():
            raise CircuitBreakerError("Circuit breaker is open")

    def is_open(self) -> bool:
        # If in open state and not yet expired
        if self._open_until and self._now() < self._open_until:
            return True
        # If open period elapsed, reset to closed
        if self._open_until and self._now() >= self._open_until:
            self._open_until = 0.0
            self._fail_count = 0
        return False

    def success(self):
        self._fail_count = 0
        self._open_until = 0.0

    def failure(self, exc: Exception | None = None):
        # Do not trip on excluded exceptions
        if exc and any(isinstance(exc, ex) for ex in self.exclude):
            return
        self._fail_count += 1
        if self._fail_count >= self.fail_max:
            self._open_until = self._now() + self.reset_timeout

# Dictionary to hold breakers per model
model_circuit_breakers = {}

# Initialize CircuitBreakers for each model
for model_key in AVAILABLE_MODELS.keys():
    # Configure breaker: 5 failures, 10s reset timeout
    model_circuit_breakers[model_key] = AsyncCircuitBreaker(
        fail_max=5,
        reset_timeout=10,
        exclude=[HTTPException]  # Do not trip breaker on expected HTTPExceptions
    )

# Configure LiteLLM
# litellm.set_verbose = True  # Removed: Deprecated
litellm.drop_params = True
litellm.max_budget = 100
# Make Langfuse callbacks optional to avoid SDK incompatibility errors by default
if os.getenv("ENABLE_LANGFUSE", "false").lower() in ("1", "true", "yes", "on"):  # opt-in
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]
# Turn on debug - Removed: Deprecated
# import litellm
# litellm._turn_on_debug()

# Ollama client for direct access
ollama_client = None

# LiteLLM Proxy async OpenAI client
openai_proxy_client: AsyncOpenAI | None = None

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
                    logger.info("✅ Ollama service is ready")
                    return True
        except Exception as e:
            logger.warning(f"⏳ Waiting for Ollama service... ({retry_count + 1}/{max_retries})")
            await asyncio.sleep(2)
            retry_count += 1
    
    logger.error("❌ Could not connect to Ollama service")
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
                    logger.info(f"📥 Downloading model {model_name}... This may take 10-30 minutes.")
                    logger.info(f"💡 Large models may require significant disk space and RAM")
                    
                    pull_data = {"name": model_name, "stream": True}
                    
                    async with client.stream("POST", f"{ollama_base_url}/api/pull", json=pull_data) as pull_response:
                        if pull_response.status_code == 200:
                            async for line in pull_response.aiter_lines():
                                if line:
                                    try:
                                        data = eval(line)
                                        if "status" in data:
                                            logger.info(f"📡 {model_name}: {data['status']}")
                                        if data.get("status") == "success":
                                            logger.info(f"✅ Model {model_name} downloaded successfully!")
                                            break
                                    except:
                                        continue
                        else:
                            logger.warning(f"⚠️ Failed to download model {model_name}. Status: {pull_response.status_code}")
                else:
                    logger.info(f"✅ Model {model_name} already available")
            
            return True
                
    except Exception as e:
        logger.error(f"❌ Error ensuring model availability: {e}")
        return True

def setup_ollama_client():
    """Setup OpenAI client pointing to Ollama"""
    global ollama_client
    
    ollama_client = OpenAI(
        base_url=f"{ollama_base_url}/v1",
        api_key="ollama"
    )
    logger.info("✅ Ollama client configured")

def setup_litellm():
    """Configure LiteLLM for Ollama"""
    # Set the Ollama base URL for LiteLLM (in-process) when not using proxy
    os.environ["OLLAMA_API_BASE"] = ollama_base_url
    os.environ["OLLAMA_API_KEY"] = "ollama"  # Added OLLAMA_API_KEY
    os.environ['LITELLM_LOG'] = 'DEBUG'  # Added: Configure LiteLLM logging via env var

    # Removed global litellm.set_timeout and litellm.max_retries as they are not direct attributes
    # These will be passed directly to acompletion calls where needed.

    logger.info(f"✅ LiteLLM configured for Ollama")


def setup_litellm_proxy_client():
    """Setup Async OpenAI client pointing to the standalone LiteLLM proxy"""
    global openai_proxy_client
    try:
        openai_proxy_client = AsyncOpenAI(
            base_url=f"{litellm_proxy_url}/v1",
            api_key=litellm_proxy_api_key
        )
        logger.info("✅ LiteLLM proxy client configured")
    except Exception as e:
        logger.error(f"❌ Failed to configure LiteLLM proxy client: {e}")
        openai_proxy_client = None


@app.get("/")
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
            "models": "/models"
        }
    }

@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{ollama_base_url}/api/tags")
            ollama_healthy = response.status_code == 200

        proxy_healthy = None
        if use_litellm_proxy:
            try:
                headers = {"Authorization": f"Bearer {litellm_proxy_api_key}"}
                async with httpx.AsyncClient() as client:
                    proxy_resp = await client.get(f"{litellm_proxy_url}/v1/models", headers=headers, timeout=5)
                    proxy_healthy = proxy_resp.status_code == 200
            except Exception as _:
                proxy_healthy = False

        return {
            "status": "healthy" if (ollama_healthy and (proxy_healthy in (None, True))) else "unhealthy",
            "ollama_service": "up" if ollama_healthy else "down",
            "litellm_proxy_enabled": use_litellm_proxy,
            "litellm_proxy_url": litellm_proxy_url,
            "litellm_proxy_service": ("up" if proxy_healthy else "down") if proxy_healthy is not None else "n/a",
            "default_model": default_model,
            "available_models": list(AVAILABLE_MODELS.keys()),
            "litellm_enabled": True
        }
    except Exception as e:
        logger.error(f"Error in /health endpoint: {e}")
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
        logger.error(f"Error fetching models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")

@app.get("/health/models")
async def health_models():
    """Check the health status of each configured model in Ollama, including response metrics."""
    model_health_statuses = {}
    for model_key, model_info in AVAILABLE_MODELS.items():
        model_name = model_info["name"]
        status_entry = {"status": "unknown"}
        
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
                        "description": model_info.get("description")
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
                test_messages = [{
                    "role": "user", 
                    "content": "Hi"
                }]
                test_start_time = time.time()
                # Use unified_completion for the test, but handle its output carefully
                test_response = await unified_completion(
                    model_name_key=model_key,
                    messages=test_messages,
                    max_tokens=5, # Keep it very short
                    temperature=0.1,
                    stream=False,
                    reasoning="none",
                    ollama_base_url=ollama_base_url
                )
                test_end_time = time.time()
                test_latency_ms = (test_end_time - test_start_time) * 1000
                
                # Check if the response contains content (indicating success)
                if hasattr(test_response, 'choices') and test_response.choices[0].message.content:
                    test_success = True
                elif isinstance(test_response, dict) and test_response.get("response"): # Direct Ollama fallback
                    test_success = True
                
                status_entry.update({
                    "litellm_test_status": "responsive",
                    "litellm_test_latency_ms": round(test_latency_ms, 2),
                    "litellm_test_success": test_success
                })
            except Exception as e:
                status_entry.update({
                    "litellm_test_status": "unresponsive",
                    "litellm_test_error": str(e),
                    "litellm_test_latency_ms": test_latency_ms, # May be partial if error occurred early
                    "litellm_test_success": False
                })
        
        # Determine overall status
        if status_entry.get("ollama_status") == "available" and status_entry.get("litellm_test_success"):
            status_entry["status"] = "healthy"
        else:
            status_entry["status"] = "unhealthy"

        model_health_statuses[model_key] = status_entry

    return JSONResponse(content=model_health_statuses)


async def unified_completion(
    model_name_key: str, # Key from AVAILABLE_MODELS, e.g., "gpt-oss:20b"
    messages: List[Dict[str, str]], # Already formatted with system prompt if needed
    max_tokens: int,
    temperature: float,
    stream: bool,
    reasoning: str, # Passed from ChatRequest, used for direct Ollama prompt
    ollama_base_url: str
):
    selected_model_info = AVAILABLE_MODELS[model_name_key]
    litellm_model = selected_model_info["litellm_name"]
    ollama_model_name = selected_model_info["name"] # Actual Ollama model name

    # Get timeout and retries from environment variables, with defaults
    litellm_timeout = int(os.getenv("LITELLM_TIMEOUT", 600)) # Default to 600 seconds (10 minutes)
    litellm_retries = int(os.getenv("LITELLM_RETRIES", 3)) # Default to 3 retries

    start_time = time.time() # Start timer
    success = False

    # Get the circuit breaker for this model
    breaker = model_circuit_breakers.get(model_name_key)
    if not breaker: # Should not happen if initialized correctly
        logger.error(f"No circuit breaker found for model: {model_name_key}")
        raise HTTPException(status_code=500, detail=f"Internal error: No circuit breaker for {model_name_key}")

    try:
        if stream:
            async def generate_stream():
                nonlocal success  # Allow modification of success in outer scope
                # Ensure the breaker measures failures happening during stream consumption
                breaker.begin()
                try:
                    primary_error = None
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
                        # Try in-process LiteLLM streaming
                        litellm_response_generator = await acompletion(
                            model=litellm_model,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            stream=True,
                            api_base=ollama_base_url,
                            timeout=litellm_timeout,  # Pass timeout here
                            num_retries=litellm_retries  # Pass retries here
                        )
                        async for chunk in litellm_response_generator:
                            yield chunk  # Yield raw LiteLLM chunk object
                        success = True  # Mark as success if stream completes without error
                except Exception as primary_error:
                    logger.error(f"Primary Streaming Path Error: {primary_error}")
                    # Fallback to direct Ollama streaming
                    try:
                        async with httpx.AsyncClient(timeout=litellm_timeout) as client:  # Use litellm_timeout for httpx
                            # For direct Ollama, we need to reconstruct the prompt
                            # Assuming the last message is the user's message
                            ollama_prompt = f"Reasoning: {reasoning}\nUser: {messages[-1]['content']}\nAssistant:"

                            ollama_request_payload = {
                                "model": ollama_model_name,
                                "prompt": ollama_prompt,
                                "options": {
                                    "num_predict": max_tokens,
                                    "temperature": temperature
                                },
                                "stream": True
                            }
                            async with client.stream("POST", f"{ollama_base_url}/api/generate", json=ollama_request_payload) as direct_response:
                                if direct_response.status_code == 200:
                                    async for line in direct_response.aiter_lines():
                                        if line:
                                            try:
                                                data = json.loads(line)
                                                yield data  # Yield raw Ollama JSON dict per line
                                                if data.get("done"):  # Check for done flag from Ollama direct
                                                    success = True  # Mark as success if direct stream completes
                                            except json.JSONDecodeError:
                                                continue
                                else:
                                    logger.error(
                                        f"Direct Ollama Streaming Fallback Failed: {direct_response.status_code} - {await direct_response.text()}"
                                    )
                                    raise HTTPException(
                                        status_code=500,
                                        detail=f"Both primary and direct Ollama streaming failed. Status: {direct_response.status_code}"
                                    )
                    except Exception as fallback_error:
                        logger.error(f"Direct Ollama Streaming Fallback Exception: {fallback_error}")
                        # Re-raise to be caught by breaker and increment failure count
                        raise HTTPException(
                            status_code=500,
                            detail=f"All streaming methods failed: Primary: {primary_error}, Direct: {fallback_error}"
                        )
                finally:
                    end_time = time.time()  # End timer for streaming
                    latency_ms = (end_time - start_time) * 1000
                    logger.info(
                        f"METRIC: Model={model_name_key}, Type=Streaming, Latency={latency_ms:.2f}ms, Success={success}"
                    )
                    if success:
                        breaker.success()
                    else:
                        breaker.failure()
            return generate_stream()
        else:
            # Non-streaming path: guard the actual call with the breaker
            breaker.begin()
            try:
                if use_litellm_proxy:
                    if openai_proxy_client is None:
                        raise RuntimeError("LiteLLM proxy client not initialized")
                    response = await openai_proxy_client.chat.completions.create(
                        model=model_name_key,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=False,
                    )
                else:
                    # Try in-process LiteLLM non-streaming
                    response = await acompletion(
                        model=litellm_model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        api_base=ollama_base_url,
                        timeout=litellm_timeout,  # Pass timeout here
                        num_retries=litellm_retries  # Pass retries here
                    )
                success = True
                breaker.success()
                return response  # Return response object
            except Exception as primary_error:
                logger.error(f"Primary Non-Streaming Path Error: {primary_error}")
                # Fallback to direct Ollama non-streaming
                try:
                    async with httpx.AsyncClient(timeout=litellm_timeout) as client:  # Use litellm_timeout for httpx
                        # For direct Ollama, we need to reconstruct the prompt
                        ollama_prompt = f"Reasoning: {reasoning}\nUser: {messages[-1]['content']}\nAssistant:"

                        ollama_request_payload = {
                            "model": ollama_model_name,
                            "prompt": ollama_prompt,
                            "options": {
                                "num_predict": max_tokens,
                                "temperature": temperature
                            },
                            "stream": False
                        }
                        direct_response = await client.post(
                            f"{ollama_base_url}/api/generate",
                            json=ollama_request_payload
                        )
                        if direct_response.status_code == 200:
                            success = True
                            breaker.success()
                            return direct_response.json()  # Return direct Ollama JSON response
                        else:
                            logger.error(
                                f"Direct Ollama Fallback Failed: {direct_response.status_code} - {await direct_response.text()}"
                            )
                            breaker.failure()
                            raise HTTPException(
                                status_code=500,
                                detail=f"Both primary and direct Ollama failed. Status: {direct_response.status_code}"
                            )
                except Exception as fallback_error:
                    logger.error(f"Direct Ollama Fallback Exception: {fallback_error}")
                    # Mark failure and re-raise to be caught by API layer
                    breaker.failure(fallback_error)
                    raise HTTPException(
                        status_code=500,
                        detail=f"All methods failed: Primary: {primary_error}, Direct: {fallback_error}"
                    )
            finally:
                end_time = time.time()  # End timer for non-streaming
                latency_ms = (end_time - start_time) * 1000
                logger.info(
                    f"METRIC: Model={model_name_key}, Type=Non-Streaming, Latency={latency_ms:.2f}ms, Success={success}"
                )

    except CircuitBreakerError:
        logger.warning(f"Circuit breaker is OPEN for model: {model_name_key}. Skipping request.")
        raise HTTPException(status_code=503, detail=f"Model {model_name_key} is temporarily unavailable (circuit breaker is open).")
    except Exception as e:
        # Any other unexpected error will trip the breaker
        logger.error(f"Unexpected error in unified_completion for {model_name_key}: {e}")
        raise e # Re-raise to let the breaker count this as a failure


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint using LiteLLM with robust fallback"""
    selected_model = request.model if request.model and request.model in AVAILABLE_MODELS else default_model
    
    # Format message with reasoning level for unified_completion
    messages_for_completion = [
        {"role": "system", "content": f"You are a helpful assistant. Reasoning: {request.reasoning}"},
        {"role": "user", "content": request.message}
    ]
    
    try:
        completion_result = await unified_completion(
            model_name_key=selected_model,
            messages=messages_for_completion,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=request.stream,
            reasoning=request.reasoning,
            ollama_base_url=ollama_base_url
        )
        
        if request.stream:
            async def generate_formatted_stream():
                async for chunk in completion_result:
                    if hasattr(chunk, 'choices') and chunk.choices[0].delta.content:
                        yield f"data: {chunk.choices[0].delta.content}\n\n" # LiteLLM chunk
                    elif "response" in chunk: # Direct Ollama chunk
                        yield f"data: {chunk['response']}\n\n"
                    if hasattr(chunk, 'choices') and chunk.choices[0].finish_reason or chunk.get("done"): # Check for finish reason or done flag
                        yield "data: [DONE]\n\n"
            return StreamingResponse(generate_formatted_stream(), media_type="text/plain")
        else:
            # Check if it's a LiteLLM response object or direct Ollama JSON
            if hasattr(completion_result, 'choices'): # LiteLLM response
                content = completion_result.choices[0].message.content
                usage_info = {
                    "prompt_tokens": getattr(completion_result.usage, 'prompt_tokens', 0),
                    "completion_tokens": getattr(completion_result.usage, 'completion_tokens', 0),
                    "total_tokens": getattr(completion_result.usage, 'total_tokens', 0)
                } if hasattr(completion_result, 'usage') and completion_result.usage else {}
            else: # Direct Ollama JSON response
                content = completion_result.get("response", "")
                usage_info = {
                    "prompt_tokens": completion_result.get("prompt_eval_count", 0),
                    "completion_tokens": completion_result.get("eval_count", 0),
                    "total_tokens": completion_result.get("prompt_eval_count", 0) + completion_result.get("eval_eval_count", 0)
                }
            
            return ChatResponse(
                response=content,
                model=selected_model,
                usage=usage_info
            )
        
    except HTTPException as e:
        raise e # Re-raise HTTPExceptions from unified_completion
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response with model {selected_model}: {str(e)}")

# OpenAI-compatible endpoint using LiteLLM
@app.post("/v1/chat/completions")
async def litellm_chat_completions(request: OpenAIChatRequest):
    """OpenAI-compatible endpoint using LiteLLM with robust fallback"""
    try:
        # Validate model
        if request.model not in AVAILABLE_MODELS:
            raise HTTPException(
                status_code=400, 
                detail=f"Model {request.model} not available. Available models: {list(AVAILABLE_MODELS.keys())}"
            )
        
        # OpenAIChatRequest messages are already in the correct format for LiteLLM
        # For direct Ollama fallback, we'll use a default reasoning if not explicitly in messages.
        default_reasoning = "medium"
        for msg in request.messages:
            if msg["role"] == "system" and "Reasoning:" in msg["content"]:
                try:
                    default_reasoning = msg["content"].split("Reasoning:")[1].strip().split(" ")[0] # Extract first word after Reasoning:
                except IndexError:
                    pass
                break

        completion_result = await unified_completion(
            model_name_key=request.model,
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=request.stream,
            reasoning=default_reasoning, # Use extracted or default reasoning
            ollama_base_url=ollama_base_url
        )
        
        if request.stream:
            async def generate_formatted_stream():
                async for chunk in completion_result:
                    # LiteLLM chunk objects are already in a format that can be dumped
                    # Direct Ollama chunks are dicts that need to be wrapped
                    if hasattr(chunk, 'choices'): # LiteLLM chunk
                        yield f"data: {chunk.model_dump_json()}\n\n"
                    elif "response" in chunk: # Direct Ollama chunk
                        # Reconstruct a basic OpenAI-like chunk for direct Ollama fallback
                        ollama_chunk = {
                            "id": "chatcmpl-ollama-fallback",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": chunk["response"]},
                                    "finish_reason": "stop" if chunk.get("done") else None
                                }
                            ]
                        }
                        yield f"data: {json.dumps(ollama_chunk)}\n\n"
                    if chunk.get("done"): # Check for done flag from Ollama direct
                        yield "data: [DONE]\n\n"
            return StreamingResponse(generate_formatted_stream(), media_type="text/event-stream") # Changed media_type for OpenAI-compatible streaming
        else:
            # Check if it's a LiteLLM response object or direct Ollama JSON
            if hasattr(completion_result, 'choices'): # LiteLLM response
                return completion_result.model_dump()
            else: # Direct Ollama JSON response
                # Reconstruct a basic OpenAI-like response for direct Ollama fallback
                ollama_response = {
                    "id": "chatcmpl-ollama-fallback",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": completion_result.get("response", "")},
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": completion_result.get("prompt_eval_count", 0),
                        "completion_tokens": completion_result.get("eval_count", 0),
                        "total_tokens": completion_result.get("prompt_eval_count", 0) + completion_result.get("eval_eval_count", 0)
                    }
                }
                return ollama_response
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in LiteLLM endpoint: {str(e)}")

# Direct Ollama endpoint for testing (without LiteLLM) - This endpoint remains unchanged as it's direct Ollama
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
                        "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_eval_count", 0)
                    }
                )
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                raise HTTPException(status_code=500, detail=f"Ollama API error: {response.status_code}")
                
    except Exception as e:
        logger.error(f"Error with direct Ollama call: {str(e)}")
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
        logger.info(f"--- Initiating Enhanced LiteLLM debug for model: {litellm_model_name} ---")
        response_obj = await acompletion(**request_payload)
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        logger.info(f"--- LiteLLM debug successful (Latency: {latency_ms:.2f} ms) ---")

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
        logger.error(f"--- LiteLLM debug FAILED (Latency: {latency_ms:.2f} ms) ---")

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
    # The /chat endpoint now uses unified_completion, so we can just call it.
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
                    logger.warning(f"⚠️ Ollama embeddings not available, using dummy embedding for: {text[:50]}...")
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
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

if __name__ == "__main__":
    logger.info("🚀 Starting Multi-Model API Gateway with LiteLLM...")
    logger.info("🌐 Local API URL: http://localhost:8000")
    logger.info("🌐 Web UI URL: http://localhost:3000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

