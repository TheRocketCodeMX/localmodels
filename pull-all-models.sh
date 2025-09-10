#!/bin/bash

# Pull all models script
# This script downloads all the configured models for the multi-model API

echo "🚀 Downloading all configured models..."
echo "⚠️ This process may take 30+ minutes depending on your internet connection"
echo "⚠️ Make sure you have enough disk space (>100GB recommended)"

# Models to download
MODELS=(
    "gpt-oss:20b"
    "qwen3:30b" 
    "devstral"
    "gemma3"
)

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "❌ Ollama is not running on localhost:11434"
    echo "   Make sure Ollama service is started first"
    exit 1
fi

echo "✅ Ollama service detected"

# Download each model
for model in "${MODELS[@]}"; do
    echo ""
    echo "📥 Downloading $model..."
    
    # Check if model already exists
    if curl -s http://localhost:11434/api/tags | grep -q "$model"; then
        echo "✅ $model already exists, skipping"
        continue
    fi
    
    # Pull the model
    if curl -X POST http://localhost:11434/api/pull \
        -H "Content-Type: application/json" \
        -d "{\"name\":\"$model\", \"stream\": false}"; then
        echo "✅ $model downloaded successfully"
    else
        echo "❌ Failed to download $model"
        echo "   You can try downloading manually with: ollama pull $model"
    fi
done

echo ""
echo "🎉 All models download process completed!"
echo "📋 To see downloaded models run: ollama list"