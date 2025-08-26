#!/bin/bash

echo "🚀 Downloading additional AI models..."

# Function to pull model using docker exec
pull_model() {
    local model=$1
    local max_attempts=3
    local attempt=1
    
    echo "📥 Downloading model: $model (attempt $attempt/$max_attempts)"
    
    while [ $attempt -le $max_attempts ]; do
        if docker exec ollmalocally-ollama-1 ollama pull $model; then
            echo "✅ Model $model downloaded successfully!"
            return 0
        else
            echo "❌ Failed to download $model (attempt $attempt/$max_attempts)"
            attempt=$((attempt + 1))
            if [ $attempt -le $max_attempts ]; then
                echo "🔄 Retrying in 10 seconds..."
                sleep 10
            fi
        fi
    done
    
    echo "❌ Failed to download $model after $max_attempts attempts"
    return 1
}

# Check if ollama container is running
if ! docker ps | grep -q "ollmalocally-ollama-1"; then
    echo "❌ Ollama container is not running. Please start with: docker-compose up"
    exit 1
fi

# Download models in sequence
echo "📦 Starting model downloads..."

# Pull DeepSeek model
pull_model "deepseek-coder:6.7b"

# Pull Llama3 model  
pull_model "llama3:8b"

# Pull Mistral model
pull_model "mistral:7b"

# Pull original GPT-OSS model if not already present
pull_model "${GPT_OSS_MODEL:-gpt-oss:20b}"

echo "🎉 Model download process completed! Available models:"
docker exec ollmalocally-ollama-1 ollama list

echo ""
echo "💡 You can now use these models in the WebUI at http://localhost:3000"
echo "💡 To download additional models manually, use: docker exec ollmalocally-ollama-1 ollama pull <model-name>"