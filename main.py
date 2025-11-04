"""
FastAPI app for LSTM Stress Detection Model (Binary Output)
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from pathlib import Path
import os

# Load model and scaler
MODEL_PATH = "models/stress_lstm_model.keras"  # ← CHANGED FROM .h5
SCALER_PATH = "models/scaler.pkl"

print("🔹 Loading model and scaler...")
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}")
    
    model = load_model(MODEL_PATH)
    scaler = pickle.load(open(SCALER_PATH, 'rb'))
    print("✅ Model and scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model/scaler: {e}")
    model = None
    scaler = None

# Initialize FastAPI
app = FastAPI(
    title="🏥 LSTM Stress Detection API",
    description="Real-time stress detection using Bidirectional LSTM",
    version="1.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class HeartRateRespirationSequence(BaseModel):
    hr_values: list      # List of 128 HR values
    resp_values: list    # List of 128 respiration values

# Response model
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probability_score: float

# Health check
@app.get("/health", tags=["Health"])
async def health():
    """Check API health"""
    return {
        "status": "✅ API is healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

# Predict from sequence
@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(data: HeartRateRespirationSequence):
    """
    Predict stress from 128 HR + Respiration values
    
    **Input:** 
    - hr_values: List of 128 heart rate values
    - resp_values: List of 128 respiration values
    
    **Output:** Stress prediction + confidence
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded")
    
    try:
        # Validate input
        if len(data.hr_values) != 128 or len(data.resp_values) != 128:
            raise HTTPException(status_code=400, detail="Must provide exactly 128 HR and 128 respiration values")
        
        # Stack HR and respiration
        hr_array = np.array(data.hr_values).reshape(-1, 1)
        resp_array = np.array(data.resp_values).reshape(-1, 1)
        features = np.hstack([hr_array, resp_array])
        
        # Normalize using the same scaler
        features_normalized = scaler.transform(features)
        
        # Reshape for LSTM (1, 128, 2) - 1 sample, 128 timesteps, 2 features
        sequence = features_normalized.reshape(1, 128, 2)
        
        # Predict (binary output: 0-1)
        prediction_prob = float(model.predict(sequence, verbose=0)[0][0])
        
        # Determine class (threshold 0.5)
        predicted_class = "Stressed" if prediction_prob > 0.5 else "Not Stressed"
        confidence = max(prediction_prob, 1 - prediction_prob)
        
        return PredictionResponse(
            prediction=predicted_class,
            confidence=confidence,
            probability_score=prediction_prob
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Info endpoint
@app.get("/info", tags=["Info"])
async def info():
    """Get model information"""
    return {
        "model": "Bidirectional LSTM Stress Detection",
        "input_size": 128,
        "input_features": ["Heart Rate", "Respiration"],
        "classes": ["Not Stressed", "Stressed"],
        "accuracy": "~99%",
        "model_type": "Binary classification (sigmoid output)"
    }

# Root
@app.get("/", tags=["Root"])
async def root():
    """Welcome message"""
    return {
        "message": "🏥 LSTM Stress Detection API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }