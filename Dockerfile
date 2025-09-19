FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY app/ app/
COPY api/ api/
COPY logging_config.py .
COPY litellm_config.json .
# Include tests and pytest config inside the image for in-container test runs
COPY pytest.ini .
COPY tests/ tests/

EXPOSE 8000 4000

# Start via app.py to preserve existing scripts and avoid module name conflicts
CMD ["python", "app.py"]
