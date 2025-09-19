from typing import Optional, List, Dict, Any
import time
import httpx
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Local request schema to avoid circular imports with app.py
class ChatRequest(BaseModel):
    model: Optional[str] = None
    message: str
    max_tokens: int = 150
    temperature: float = 0.7
    stream: bool = False
    reasoning: str = ""


def get_router(AVAILABLE_MODELS: Dict[str, Dict[str, Any]], ollama_base_url: str, logger) -> APIRouter:
    """
    Build and return the APIRouter exposing the debug endpoint.

    Dependencies are passed in to avoid circular imports with app.py
    and to keep this router self-contained.
    """
    router = APIRouter()

    @router.post("/debug/litellm/{model_name}", include_in_schema=False)
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
        hardware_info: Dict[str, Any] = {}
        warning_message: Optional[str] = None

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

                    # Guess which model
                    matched_model = None
                    try:
                        for m in ps_models:
                            mn = _norm_local(m.get("name")) if isinstance(m, dict) else ""
                            if mn and (mn == norm_target or norm_target.endswith(mn) or mn.endswith(norm_target)):
                                matched_model = m
                                break
                    except Exception:
                        pass

                    # Pull VRAM usage from any place we find it
                    def _walk(obj):
                        if isinstance(obj, dict):
                            for k, v in obj.items():
                                lk = str(k).lower()
                                if lk in ["gpu_memory", "vram_used", "vram", "gpu_memory_used", "gpu_vram_used"]:
                                    try:
                                        val = float(v)
                                        return val
                                    except Exception:
                                        pass
                                res = _walk(v)
                                if res is not None:
                                    return res
                        elif isinstance(obj, list):
                            for it in obj:
                                res = _walk(it)
                                if res is not None:
                                    return res
                        return None

                    vram_used = _walk(ps_json)

                    # Heuristics for GPU library type (prefer exact fields if present)
                    gpu_backend = None
                    try:
                        # prefer devices info if available
                        if gpu_devices:
                            libs = [d.get("library") for d in gpu_devices if isinstance(d, dict)]
                            libs = [l for l in libs if isinstance(l, str) and l]
                            if libs:
                                # choose most frequent
                                from collections import Counter
                                gpu_backend = Counter(libs).most_common(1)[0][0]
                        if not gpu_backend:
                            # fallback to model details
                            if isinstance(details_section, dict):
                                gpu_backend = (details_section.get("gpu_library") or details_section.get("llm_library") or details_section.get("library") or "").lower()
                            if not gpu_backend:
                                gpu_backend = library
                        if gpu_backend in (None, "", "unknown"):
                            gpu_backend = None
                    except Exception:
                        gpu_backend = None

                    # Legion of small hints to get a user-friendly GPU name
                    def _probe_system_gpu_name(lib_hint: str | None) -> Optional[str]:
                        try:
                            # 1) prefer exact device names from ps_json
                            if gpu_devices:
                                parts = []
                                for d in gpu_devices:
                                    nm = d.get("name") if isinstance(d, dict) else None
                                    if isinstance(nm, str) and nm.strip():
                                        parts.append(nm.strip())
                                if parts:
                                    return ", ".join(sorted(set(parts)))

                            # 2) probe libraries if available
                            if lib_hint == "rocm":
                                try:
                                    import torch
                                    if getattr(torch.version, "hip", None):
                                        amd_name = None
                                        try:
                                            import subprocess, json as _json
                                            out = subprocess.check_output(["bash", "-lc", "rocminfo | grep -i 'Name:' | head -n 1 | awk '{print $2}'"], stderr=subprocess.DEVNULL, timeout=2)
                                            amd_name = out.decode().strip()
                                        except Exception:
                                            pass

                                        if not amd_name:
                                            try:
                                                import pynvml  # type: ignore
                                                # unlikely on ROCm-only systems, skip
                                            except Exception:
                                                pass
                                        return amd_name or "AMD GPU (ROCm)"
                                except Exception:
                                    pass

                            if lib_hint == "cuda":
                                try:
                                    import pynvml  # type: ignore
                                    pynvml.nvmlInit()
                                    count = pynvml.nvmlDeviceGetCount()
                                    names = []
                                    for i in range(count):
                                        h = pynvml.nvmlDeviceGetHandleByIndex(i)
                                        names.append(pynvml.nvmlDeviceGetName(h).decode())
                                    pynvml.nvmlShutdown()
                                    if names:
                                        return ", ".join(names)
                                except Exception:
                                    pass

                            if lib_hint == "metal":
                                return "Apple GPU (Metal)"

                            return None
                        except Exception:
                            return None

                    gpu_name = _probe_system_gpu_name(gpu_backend)

                    # VRAM from details if present
                    size_vram_bytes = None
                    try:
                        size_vram_bytes = details_section.get("size_vram") or details_section.get("size_vram_bytes")
                    except Exception:
                        size_vram_bytes = None

                    # Construct curated info
                    def pick_first(*vals):
                        for v in vals:
                            if v not in (None, "", [], {}):
                                return v
                        return None

                    info = {
                        "model": ollama_model_name,
                        "library": pick_first(details_section.get("gpu_library"), details_section.get("llm_library"), library, gpu_backend) or "unknown",
                        "gpu_model": gpu_name or "N/A",
                        "gpu_layers_offloaded": pick_first(details_section.get("gpu_layers_offloaded"), details_section.get("gpu_layers"), details_section.get("num_gpu")),
                        "vram_used_gb": _bytes_to_gb(vram_used),
                        "size_vram_bytes": size_vram_bytes,
                        "devices": gpu_devices,
                        "raw_endpoints": {
                            "show": f"{ollama_base_url}/api/show",
                            "ps": f"{ollama_base_url}/api/ps",
                        },
                        "raw_status": {
                            "show": show_resp.status_code if 'show_resp' in locals() else None,
                            "ps": ps_resp.status_code if 'ps_resp' in locals() else None,
                        }
                    }

                    # Provide hint if no matching model is found in ps
                    warn = None
                    try:
                        if not matched_model:
                            if ps_models:
                                warn = (
                                    "Model process not found yet. The model may still be loading/running. "
                                    "Try again after invoking a generation or wait a few seconds."
                                )
                        else:
                            # If present but no devices detected, hint to check GPU setup
                            if not gpu_devices:
                                warn = (
                                    "Model is loaded but no GPU devices detected in /api/ps. "
                                    "It might be running on CPU or the backend isn't exposing GPU info."
                                )
                    except Exception:
                        pass

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
            from litellm import acompletion
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
                "status": "ok",
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

    return router
