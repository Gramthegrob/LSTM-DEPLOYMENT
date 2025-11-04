"""
Run FastAPI server locally with ngrok tunnel
"""
import uvicorn
from pyngrok import ngrok
import asyncio

# Set ngrok auth token (optional but recommended)
# ngrok.set_auth_token("YOUR_NGROK_TOKEN")

def start_server():
    """Start FastAPI server with ngrok tunnel"""
    
    # Start ngrok tunnel
    print("\n" + "="*60)
    print("🌐 Starting ngrok tunnel...")
    print("="*60)
    
    public_url = ngrok.connect(8000)
    print(f"\n✅ ngrok tunnel created!")
    print(f"🔗 Public URL: {public_url}")
    
    # Print API endpoints
    print("\n" + "="*60)
    print("🚀 API ENDPOINTS")
    print("="*60)
    print(f"📚 Docs (Swagger):     {public_url}/docs")
    print(f"📚 ReDoc:              {public_url}/redoc")
    print(f"🏥 Health:             {public_url}/health")
    print(f"🔮 Predict:            {public_url}/predict")
    print(f"ℹ️ Info:               {public_url}/info")
    print("="*60)
    
    # Start server
    print("\n▶️ Starting FastAPI server on http://localhost:8000\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    start_server()