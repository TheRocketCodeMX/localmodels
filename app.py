# Thin compatibility shim to keep using app.py as entrypoint
# while the actual application lives in the modular package app/main.py

from app.main import app  # exposes the FastAPI instance

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

