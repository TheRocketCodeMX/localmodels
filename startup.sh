#!/bin/bash

echo "🚀 Starting GPT-OSS Gateway with Ollama..."

# Wait a bit for Ollama service to be ready
sleep 5

# Start the API server
python app.py