# GPT-OSS con Ollama + Docker 

Esta es la versión **optimizada** usando Ollama para ejecutar GPT-OSS localmente con APIs estructuradas y compatibles con OpenAI.

## ¿Por qué Ollama es mejor?

| Ventaja | Descripción |
|---------|-------------|
| **APIs REST nativas** | Compatible 100% con OpenAI API |
| **Menos memoria** | Modelos cuantizados automáticamente |
| **Gestión simplificada** | Auto-descarga y gestión de modelos |
| **Mejor performance** | Optimizado para hardware consumer |
| **Múltiples interfaces** | REST, OpenAI SDK, WebUI |

## Configuración Rápida

1. **Configura variables de entorno:**
   ```bash
   cp .env.example .env
   # Edita .env con tu token de ngrok
   ```

2. **Ejecuta con Docker Compose:**
   ```bash
   docker-compose up --build
   ```

## Servicios Incluidos

- **Ollama** (Puerto 11434): Motor de IA
- **API Gateway** (Puerto 8000): Tu API personalizada + ngrok
- **Web UI** (Puerto 3000): Interfaz web para testing

## Endpoints Disponibles

### 1. API Personalizada (Puerto 8000)
```bash
# Chat básico
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explícame quantum computing",
    "max_tokens": 300,
    "temperature": 0.7,
    "reasoning": "high"
  }'

# Información del sistema
curl http://localhost:8000/

# Estado de salud
curl http://localhost:8000/health

# Listar modelos
curl http://localhost:8000/models
```

### 2. OpenAI Compatible (Puerto 8000)
```python
from openai import OpenAI

# Usa tu endpoint público de ngrok o local
client = OpenAI(
    base_url="http://localhost:8000/v1",  # O tu URL de ngrok
    api_key="dummy"
)

response = client.chat.completions.create(
    model="gpt-oss:20b",
    messages=[
        {"role": "system", "content": "Eres un experto en IA."},
        {"role": "user", "content": "¿Qué es GPT-OSS?"}
    ],
    max_tokens=200,
    temperature=0.7
)

print(response.choices[0].message.content)
```

### 3. Ollama Directo (Puerto 11434)
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

response = client.chat.completions.create(
    model="gpt-oss:20b",
    messages=[{"role": "user", "content": "Hola"}]
)
```

## Modelos Disponibles

- **gpt-oss:20b** - 21B parámetros (~16GB RAM)
- **gpt-oss:120b** - 117B parámetros (~80GB RAM)

Los modelos se descargan automáticamente la primera vez.

## Comandos Útiles

```bash
# Ver logs en tiempo real
docker-compose logs -f

# Solo Ollama sin gateway
docker run -d -v ollama:/root/.ollama -p 11434:11434 ollama/ollama

# Descargar modelo manualmente
docker exec -it <ollama_container> ollama pull gpt-oss:20b

# Listar modelos descargados
docker exec -it <ollama_container> ollama list
```

## Testing con Web UI

Accede a `http://localhost:3000` para una interfaz web completa donde puedes:
- Chatear con GPT-OSS
- Ajustar parámetros
- Ver estadísticas de uso
- Probar diferentes prompts

## Ventajas vs. Implementación Directa

✅ **APIs estructuradas y estándar**  
✅ **Compatible con cualquier cliente OpenAI**  
✅ **Menor uso de memoria (cuantización automática)**  
✅ **Gestión automática de modelos**  
✅ **Múltiples interfaces (REST, SDK, Web)**  
✅ **Mejor optimización de hardware**  
✅ **Streaming nativo**  
✅ **Métricas y monitoring integrado**

## URLs Finales

Cuando ejecutes el stack completo tendrás:
- **API Local**: `http://localhost:8000`
- **Ollama Directo**: `http://localhost:11434`
- **Web UI**: `http://localhost:3000`
- **Ngrok Público**: Se muestra en los logs

¡Perfecto para desarrollo y producción! 🚀
