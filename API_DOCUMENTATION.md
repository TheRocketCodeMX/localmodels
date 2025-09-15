# LocalModels API Documentation

## Overview
Esta documentación cubre todos los endpoints disponibles en el sistema LocalModels, que incluye:
- **API Gateway personalizado** (Puerto 8000)
- **Ollama directo** (Puerto 11434) 
- **Web UI** (Puerto 3000)

## URLs Base
- **API Gateway**: `http://localhost:8000`
- **Ollama Directo**: `http://localhost:11434`
- **Web UI**: `http://localhost:3000`

---

## 1. API Gateway Personalizado (Puerto 8000)

### 1.1 Información del Sistema

#### GET `/`
Información general del sistema y endpoints disponibles.

**Ejemplo:**
```bash
curl http://localhost:8000/
```

**Respuesta:**
```json
{
  "message": "Multi-Model API with LiteLLM is running",
  "default_model": "gpt-oss:20b",
  "available_models": {
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
  },
  "ollama_url": "http://ollama:11434",
  "endpoints": {
    "chat": "/chat",
    "chat_with_model": "/chat/{model_name}",
    "chat_litellm": "/v1/chat/completions",
    "embeddings": "/v1/embeddings",
    "health": "/health",
    "models": "/models"
  }
}
```

### 1.2 Salud del Sistema

#### GET `/health`
Verificar estado de los servicios.

**Ejemplo:**
```bash
curl http://localhost:8000/health
```

**Respuesta:**
```json
{
  "status": "healthy",
  "ollama_service": "up",
  "default_model": "gpt-oss:20b",
  "available_models": ["gpt-oss:20b", "qwen3:30b", "devstral", "gemma3"],
  "litellm_enabled": true
}
```

### 1.3 Listar Modelos

#### GET `/models`
Lista todos los modelos disponibles en Ollama.

**Ejemplo:**
```bash
curl http://localhost:8000/models
```

**Respuesta:**
```json
{
  "models": [
    {
      "name": "gpt-oss:20b",
      "model": "gpt-oss:20b",
      "modified_at": "2025-09-10T17:15:32.123Z",
      "size": 17280000000,
      "digest": "sha256-b112e727c6f18875636c56a779790a590d705aec9e1c0eb5a97d51fc2a778583",
      "details": {
        "parent_model": "",
        "format": "gguf",
        "family": "gptoss",
        "families": ["gptoss"],
        "parameter_size": "20.8B",
        "quantization_level": "Q4_K_M"
      }
    }
  ]
}
```

### 1.4 Chat Endpoints

#### POST `/chat`
Endpoint principal para chat con LiteLLM.

**Parámetros:**
```json
{
  "message": "string (requerido)",
  "model": "string (opcional, por defecto: gpt-oss:20b)",
  "max_tokens": "integer (opcional, por defecto: 512)",
  "temperature": "float (opcional, por defecto: 0.7)",
  "reasoning": "string (opcional: low|medium|high, por defecto: medium)",
  "stream": "boolean (opcional, por defecto: false)"
}
```

**Ejemplos:**

**Chat básico:**
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explica qué es la inteligencia artificial",
    "max_tokens": 200,
    "temperature": 0.7,
    "reasoning": "high"
  }'
```

**Chat con modelo específico:**
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "¿Cómo programar en Python?",
    "model": "devstral",
    "max_tokens": 300,
    "temperature": 0.8
  }'
```

**Chat con streaming:**
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Cuéntame sobre el machine learning",
    "model": "gpt-oss:20b",
    "stream": true
  }'
```

**Respuesta (sin streaming):**
```json
{
  "response": "La inteligencia artificial es una rama de la informática...",
  "model": "gpt-oss:20b",
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 45,
    "total_tokens": 57
  }
}
```

#### POST `/chat/{model_name}`
Chat con un modelo específico por URL.

**Modelos disponibles:**
- `/chat/gpt-oss`
- `/chat/qwen3`
- `/chat/devstral`
- `/chat/gemma3`

**Ejemplo:**
```bash
curl -X POST "http://localhost:8000/chat/devstral" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "¿Cuál es la diferencia entre Python y JavaScript?",
    "max_tokens": 250
  }'
```

### 1.5 OpenAI Compatible Endpoint

#### POST `/v1/chat/completions`
Endpoint completamente compatible con OpenAI API.

**Parámetros:**
```json
{
  "model": "string (requerido)",
  "messages": [
    {
      "role": "system|user|assistant",
      "content": "string"
    }
  ],
  "max_tokens": "integer (opcional)",
  "temperature": "float (opcional)",
  "stream": "boolean (opcional)"
}
```

**Ejemplo:**
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss:20b",
    "messages": [
      {"role": "system", "content": "Eres un experto en programación."},
      {"role": "user", "content": "¿Cómo crear una función en Python?"}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

**Respuesta:**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1699000000,
  "model": "gpt-oss:20b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Para crear una función en Python, usa la palabra clave 'def'..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 50,
    "total_tokens": 75
  }
}
```

### 1.6 Embeddings

#### POST `/v1/embeddings`
Generar embeddings de texto.

**Parámetros:**
```json
{
  "input": "string o array de strings",
  "model": "string (opcional)",
  "encoding_format": "string (opcional: float)",
  "dimensions": "integer (opcional)"
}
```

**Ejemplo:**
```bash
curl -X POST "http://localhost:8000/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Este es un texto de ejemplo",
    "model": "gpt-oss:20b"
  }'
```

**Respuesta:**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.1, -0.2, 0.3, ...]
    }
  ],
  "model": "gpt-oss:20b",
  "usage": {
    "prompt_tokens": 6,
    "total_tokens": 6
  }
}
```

---

## 2. Ollama Directo (Puerto 11434)

### 2.1 Información de Modelos

#### GET `/api/tags`
Lista todos los modelos instalados.

**Ejemplo:**
```bash
curl http://localhost:11434/api/tags
```

#### GET `/api/ps`
Lista modelos actualmente cargados en memoria.

**Ejemplo:**
```bash
curl http://localhost:11434/api/ps
```

#### GET `/api/version`
Versión de Ollama.

**Ejemplo:**
```bash
curl http://localhost:11434/api/version
```

### 2.2 Gestión de Modelos

#### POST `/api/pull`
Descargar un modelo.

**Ejemplo:**
```bash
curl -X POST http://localhost:11434/api/pull \
  -H "Content-Type: application/json" \
  -d '{
    "name": "gemma3:latest",
    "stream": false
  }'
```

#### DELETE `/api/delete`
Eliminar un modelo.

**Ejemplo:**
```bash
curl -X DELETE http://localhost:11434/api/delete \
  -H "Content-Type: application/json" \
  -d '{
    "name": "gemma3:latest"
  }'
```

### 2.3 Chat con Ollama

#### POST `/api/chat`
Chat directo con Ollama.

**Ejemplo:**
```bash
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss:20b",
    "messages": [
      {
        "role": "user",
        "content": "¿Por qué el cielo es azul?"
      }
    ],
    "stream": false
  }'
```

#### POST `/api/generate`
Generación de texto simple.

**Ejemplo:**
```bash
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss:20b",
    "prompt": "Explica la teoría de la relatividad",
    "stream": false
  }'
```

### 2.4 OpenAI Compatible (Ollama)

#### POST `/v1/chat/completions`
Endpoint OpenAI directo de Ollama.

**Ejemplo:**
```bash
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss:20b",
    "messages": [
      {"role": "user", "content": "Hola"}
    ]
  }'
```

---

## 3. Uso con SDKs

### 3.1 Python con OpenAI SDK

```python
from openai import OpenAI

# Usando API Gateway
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="gpt-oss:20b",
    messages=[
        {"role": "system", "content": "Eres un asistente útil."},
        {"role": "user", "content": "¿Qué es Python?"}
    ],
    max_tokens=200,
    temperature=0.7
)

print(response.choices[0].message.content)
```

### 3.2 Python con requests

```python
import requests

# Chat básico
response = requests.post(
    "http://localhost:8000/chat",
    json={
        "message": "Explica el machine learning",
        "model": "devstral",
        "max_tokens": 300,
        "temperature": 0.8
    }
)

result = response.json()
print(result["response"])
```

### 3.3 JavaScript/Node.js

```javascript
// Usando fetch
const response = await fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    message: '¿Cómo funciona una red neuronal?',
    model: 'gpt-oss:20b',
    max_tokens: 250
  })
});

const data = await response.json();
console.log(data.response);
```

### 3.4 cURL Scripts

**Script para testing rápido:**
```bash
#!/bin/bash
# test_models.sh

MODELS=("gpt-oss:20b" "devstral" "gemma3")
BASE_URL="http://localhost:8000"

for model in "${MODELS[@]}"; do
    echo "Testing $model..."
    curl -X POST "$BASE_URL/chat" \
        -H "Content-Type: application/json" \
        -d "{
            \"message\": \"Hola, soy $model\",
            \"model\": \"$model\",
            \"max_tokens\": 50
        }" \
        | jq '.response'
    echo "---"
done
```

---

## 4. Códigos de Estado HTTP

| Código | Descripción |
|--------|-------------|
| 200 | Éxito |
| 400 | Error en parámetros |
| 500 | Error interno del servidor |
| 503 | Servicio no disponible |

## 5. Manejo de Errores

**Ejemplo de respuesta de error:**
```json
{
  "detail": "Error generating response with LiteLLM model devstral: Model requires more system memory (18.4 GiB) than is available (17.0 GiB)"
}
```

## 6. Límites y Consideraciones

### Memoria RAM requerida por modelo:
- **gpt-oss:20b**: ~16GB RAM
- **qwen3:30b**: ~24GB RAM  
- **devstral**: ~14GB RAM
- **gemma3**: ~12GB RAM

### Timeouts:
- Chat requests: 1800 segundos
- Model loading: Variable según el modelo
- Streaming: Sin timeout específico

---

## 7. Ejemplos de Integración Completa

### 7.1 Sistema de Chat Simple

```python
import requests
import json

class LocalModelsClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def chat(self, message, model="gpt-oss:20b", **kwargs):
        payload = {
            "message": message,
            "model": model,
            **kwargs
        }
        
        response = requests.post(
            f"{self.base_url}/chat",
            json=payload
        )
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            raise Exception(f"Error: {response.json()}")
    
    def get_models(self):
        response = requests.get(f"{self.base_url}/models")
        return response.json()["models"]

# Uso
client = LocalModelsClient()
response = client.chat("¿Qué es la inteligencia artificial?")
print(response)
```

### 7.2 Comparación entre Modelos

```python
def compare_models(question, models=None):
    if models is None:
        models = ["gpt-oss:20b", "devstral", "gemma3"]
    
    client = LocalModelsClient()
    results = {}
    
    for model in models:
        try:
            response = client.chat(
                question, 
                model=model, 
                max_tokens=200
            )
            results[model] = {
                "status": "success",
                "response": response
            }
        except Exception as e:
            results[model] = {
                "status": "error",
                "error": str(e)
            }
    
    return results

# Uso
question = "Explica qué es un algoritmo"
comparison = compare_models(question)
for model, result in comparison.items():
    print(f"\n{model}: {result['status']}")
    if result['status'] == 'success':
        print(result['response'][:100] + "...")
```

---

Esta documentación cubre todos los endpoints y formas de uso del sistema LocalModels. Cada endpoint incluye ejemplos prácticos que el equipo de desarrollo puede usar directamente.