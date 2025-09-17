# :clipboard: Plan de Acción - Proyecto LocalModels

Este documento consolida las acciones y pasos a seguir, manteniendo la terminología del contexto original.

---

## :martillo_y_llave_inglesa: PLAN DE SOLUCIÓN - LITELLM

### Fase 1: Corrección Inmediata
- [x] Actualizar modelo por defecto en .env a `DEFAULT_MODEL=qwen3:30b`
- [x] Rebuild contenedor con cambios: `docker-compose down`, `docker-compose build --no-cache`, `docker-compose up -d`

### Fase 2: Configuración LiteLLM
- [x] Configuración robusta en `app.py`:
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
- [ ] Métricas de latencia y success rate

---

## :portapapeles: ACCIONES REQUERIDAS (Desarrollador debe implementar)

- [x] Investigar `gpt-oss:20b`: Por qué genera respuestas vacías
- [ ] Configurar LiteLLM proxy standalone: Separar del FastAPI main
- [ ] Implementar circuit breaker: Para modelos que fallan
- [ ] Añadir logging estructurado: Con observabilidad completa
- [ ] Tests automatizados: Para cada modelo y endpoint

---

## :dardo: PRÓXIMOS PASOS INMEDIATOS (Consolidado)

- [x] Configurar LiteLLM con timeout y retry policies
- [x] Investigar incompatibilidad específica con `gpt-oss:20b`
- [ ] Documentar formato de respuesta esperado vs actual

---

## :mag_right: Archivos Críticos para Revisar

- [x] `app.py:278-321` - Lógica fallback LiteLLM
- [x] `app.py:49-54` - Configuración LiteLLM
- [x] `docker-compose.yml:30-31` - Variables de entorno
- [x] `.env:5` - Modelo por defecto
