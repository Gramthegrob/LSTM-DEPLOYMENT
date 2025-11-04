"""
Run FastAPI server (without ngrok)
Good for production servers or testing locally
"""
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("DEBUG", "False") == "True"
    
    print(f"🚀 Starting FastAPI server")
    print(f"📍 Host: {host}:{port}")
    print(f"📚 Docs: http://localhost:{port}/docs")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )