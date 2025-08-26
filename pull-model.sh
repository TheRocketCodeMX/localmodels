#!/bin/bash

# Script to manually pull GPT-OSS model
MODEL=${1:-gpt-oss:20b}

echo "🚀 Pulling model: $MODEL"
echo "💡 This may take 10-20 minutes depending on your internet speed"

# Check if ollama container is running
if ! docker ps | grep -q "ollama"; then
    echo "❌ Ollama container is not running. Please start with: docker-compose up"
    exit 1
fi

# Pull the model
echo "📥 Starting download..."
docker exec -it ollmalocally-ollama-1 ollama pull $MODEL

if [ $? -eq 0 ]; then
    echo "✅ Model $MODEL downloaded successfully!"
    echo "🎉 You can now use the model in the WebUI at http://localhost:3000"
else
    echo "❌ Failed to download model $MODEL"
fi

# List available models
echo ""
echo "📋 Available models:"
docker exec -it ollmalocally-ollama-1 ollama list