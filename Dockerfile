FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY logging_config.py .
COPY litellm_config.json .

EXPOSE 8000 4000

# Use python directly instead of shell script
CMD ["python", "app.py"]
