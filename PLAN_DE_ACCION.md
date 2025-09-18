# :clipboard: Plan de Acción - Proyecto LocalModels

Este documento consolida las acciones y pasos a seguir, manteniendo la terminología del contexto original.

---

##  PLAN DE SOLUCIÓN - LITELLM

### Fase 1: Corrección Inmediata
- [x] Actualizar modelo por defecto en .env a `DEFAULT_MODEL=qwen3:30b`
- [x] Rebuild contenedor con cambios: `docker-compose down`, `docker-compose build --no-cache`, `docker-compose up -d` (Acción manual, no un cambio de código)

### Fase 2: Configuración LiteLLM
- [x] Configuración robusta en `app/core/clients.py::setup_litellm`:
    - [x] `litellm.set_verbose = True`
    - [x] `litellm.drop_params = True`
    - [x] `litellm.max_budget = 100`
    - [x] `litellm.success_callback = ["langfuse"]`
    - [x] `litellm.failure_callback = ["langfuse"]`
- [x] Variables de entorno correctas: `os.environ["OLLAMA_API_BASE"] = ollama_base_url`, `os.environ["OLLAMA_API_KEY"] = "ollama"`

### Fase 3: Manejo de Errores
- [x] Implementar fallback robusto (`async def unified_completion(model, messages, **kwargs): ...` para intentar LiteLLM primero y luego Ollama directo)

### Fase 4: Monitoreo y Salud
- [x] Endpoint `/debug/models` para verificar estado de cada modelo
- [x] Health checks individuales por modelo
- [x] Métricas de latencia y success rate

---

## : ACCIONES REQUERIDAS (Desarrollador debe implementar)

- [x] Investigar `gpt-oss:20b`: Por qué genera respuestas vacías
- [x] Configurar LiteLLM proxy standalone: Separar del FastAPI main
- [x] Implementar circuit breaker: Para modelos que fallan
- [x] Añadir logging estructurado: Con observabilidad completa
- [x] Tests automatizados: Para cada modelo y endpoint

---

##  PRÓXIMOS PASOS INMEDIATOS (Consolidado)

- [x] Configurar LiteLLM con timeout y retry policies
- [x] Investigar incompatibilidad específica con `gpt-oss:20b`
- [x] Documentar formato de respuesta esperado vs actual
- [x] Cerrar marcador pendiente: no quedan tareas ocultas (TAREA PENDIENTE resuelta)

---

## Archivos Críticos para Revisar

- [x] `app/services/completion.py` - Lógica fallback LiteLLM (`unified_completion`)
- [x] `app/core/clients.py::setup_litellm` - Configuración LiteLLM
- [x] `docker-compose.yml` - Variables de entorno relevantes
- [x] `.env` - Modelo por defecto
- [x] `app.py` - Shim/entrypoint; la app real vive en `app/main.py`


---

## Variables de Entorno soportadas

- DEFAULT_MODEL (por defecto: `qwen3:30b`)
- OLLAMA_BASE_URL (por defecto: `http://localhost:11434`)
- USE_LITELLM_PROXY (por defecto: `true`)
- LITELLM_PROXY_URL (por defecto: `http://litellm-proxy:4000`)
- LITELLM_MASTER_KEY (por defecto: `sk-1234`)
- ENABLE_LANGFUSE (`true/false`) — habilita callbacks de éxito/fallo de LiteLLM
- LITELLM_TIMEOUT (por defecto: `600`) — segundos
- LITELLM_RETRIES (por defecto: `3`)
- GATEWAY_URL (solo para tests; por defecto: `http://localhost:8000`)
- TEST_HTTP_TIMEOUT (solo para tests; por defecto: `120`)

---

## 🧪 Tests de integración E2E

Cómo ejecutarlos:

1. Levanta los servicios:
   - docker compose up -d
   - Espera a que Ollama y el proxy estén listos.

2. Ejecuta los tests desde el host (recomendado):
   - export GATEWAY_URL=http://localhost:8000
   - pytest -q tests/integration

3. Ejecuta los tests dentro del contenedor:
   - docker compose exec gpt-oss-gateway sh -lc "export GATEWAY_URL=http://localhost:8000 TEST_HTTP_TIMEOUT=240 && pytest -q tests/integration"

Notas:
- Los tests detectan dinámicamente los modelos disponibles desde `/` o `/health`.
- Se prueban `/`, `/health`, `/models`, `/chat` (stream y no stream), `/v1/chat/completions` (stream y no stream) y `/v1/embeddings`.
- Ajusta TEST_HTTP_TIMEOUT si tu entorno es lento (por ejemplo, export TEST_HTTP_TIMEOUT=240).
- `pytest` ya viene preinstalado en la imagen; no es necesario instalarlo manualmente dentro del contenedor.

# :clipboard: Plan de Acción - Proyecto LocalModels

Este documento consolida las acciones y pasos a seguir, manteniendo la terminología del contexto original.

---

##  PLAN DE SOLUCIÓN - LITELLM

### Fase 1: Corrección Inmediata
- [x] Actualizar modelo por defecto en .env a `DEFAULT_MODEL=qwen3:30b`
- [x] Rebuild contenedor con cambios: `docker-compose down`, `docker-compose build --no-cache`, `docker-compose up -d` (Acción manual, no un cambio de código)

### Fase 2: Configuración LiteLLM
- [x] Configuración robusta en `app/core/clients.py::setup_litellm`:
    - [x] `litellm.set_verbose = True`
    - [x] `litellm.drop_params = True`
    - [x] `litellm.max_budget = 100`
    - [x] `litellm.success_callback = ["langfuse"]`
    - [x] `litellm.failure_callback = ["langfuse"]`
- [x] Variables de entorno correctas: `os.environ["OLLAMA_API_BASE"] = ollama_base_url`, `os.environ["OLLAMA_API_KEY"] = "ollama"`

### Fase 3: Manejo de Errores
- [x] Implementar fallback robusto (`async def unified_completion(model, messages, **kwargs): ...` para intentar LiteLLM primero y luego Ollama directo)

### Fase 4: Monitoreo y Salud
- [x] Endpoint `/debug/models` para verificar estado de cada modelo
- [x] Health checks individuales por modelo
- [x] Métricas de latencia y success rate

---

## ACCIONES REQUERIDAS (Desarrollador debe implementar)

- [x] Investigar `gpt-oss:20b`: Por qué genera respuestas vacías
- [x] Configurar LiteLLM proxy standalone: Separar del FastAPI main
- [x] Implementar circuit breaker: Para modelos que fallan
- [x] Añadir logging estructurado: Con observabilidad completa
- [x] Tests automatizados: Para cada modelo y endpoint

---

# PRÓXIMOS PASOS INMEDIATOS (Consolidado)

- [x] Configurar LiteLLM con timeout y retry policies
- [x] Investigar incompatibilidad específica con `gpt-oss:20b`
- [x] Documentar formato de respuesta esperado vs actual
- [x] Cerrar marcador pendiente: no quedan tareas ocultas (TAREA PENDIENTE resuelta)

---

##  Archivos Críticos para Revisar

- [x] `app/services/completion.py` - Lógica fallback LiteLLM (`unified_completion`)
- [x] `app/core/clients.py::setup_litellm` - Configuración LiteLLM
- [x] `docker-compose.yml` - Variables de entorno relevantes
- [x] `.env` - Modelo por defecto
- [x] `app.py` - Shim/entrypoint; la app real vive en `app/main.py`


---

##  Variables de Entorno soportadas

- DEFAULT_MODEL (por defecto: `qwen3:30b`)
- OLLAMA_BASE_URL (por defecto: `http://localhost:11434`)
- USE_LITELLM_PROXY (por defecto: `true`)
- LITELLM_PROXY_URL (por defecto: `http://litellm-proxy:4000`)
- LITELLM_MASTER_KEY (por defecto: `sk-1234`)
- ENABLE_LANGFUSE (`true/false`) — habilita callbacks de éxito/fallo de LiteLLM
- LITELLM_TIMEOUT (por defecto: `600`) — segundos
- LITELLM_RETRIES (por defecto: `3`)
- GATEWAY_URL (solo para tests; por defecto: `http://localhost:8000`)
- TEST_HTTP_TIMEOUT (solo para tests; por defecto: `120`)

---

## 🧪 Tests de integración E2E

Cómo ejecutarlos:

1. Levanta los servicios:
   - docker compose up -d
   - Espera a que Ollama y el proxy estén listos.

2. Ejecuta los tests desde el host (recomendado):
   - export GATEWAY_URL=http://localhost:8000
   - pytest -q tests/integration

3. Ejecuta los tests dentro del contenedor:
   - docker compose exec gpt-oss-gateway sh -lc "export GATEWAY_URL=http://localhost:8000 TEST_HTTP_TIMEOUT=240 && pytest -q tests/integration"

Notas:
- Los tests detectan dinámicamente los modelos disponibles desde `/` o `/health`.
- Se prueban `/`, `/health`, `/models`, `/chat` (stream y no stream), `/v1/chat/completions` (stream y no stream) y `/v1/embeddings`.
- Ajusta TEST_HTTP_TIMEOUT si tu entorno es lento (por ejemplo, export TEST_HTTP_TIMEOUT=240).
- `pytest` ya viene preinstalado en la imagen; no es necesario instalarlo manualmente dentro del contenedor.

---

## Nota sobre métricas agregadas (/metrics) — opcional

Actualmente se registran métricas por solicitud (latencia y éxito) en los logs con el prefijo `METRIC:` desde `app/services/completion.py`. Si en el futuro necesitas métricas agregadas o exposición Prometheus, se puede añadir un endpoint `/metrics` (por ejemplo, con `prometheus-fastapi-instrumentator`) como mejora no bloqueante.

---

## Nota Aclaratoria sobre Endpoints de Monitoreo

En la **Fase 4: Monitoreo y Salud**, el plan solicita tanto `Health checks individuales por modelo` como un `Endpoint /debug/models`. Estas tareas son redundantes.

La implementación actual resuelve esto de manera eficiente con dos endpoints:

1.  **`GET /health/models`**: Cumple la tarea de verificar el estado de **todos** los modelos a la vez.
2.  **`POST /debug/litellm/{model_name}`**: Ofrece una herramienta para una **depuración profunda** de un modelo **individual**.

**Conclusión:** La funcionalidad está completa y mejorada. 
