#!/usr/bin/env python3
"""
FastAPI ML Inference Service for Wine Classification
Deployable on Kubernetes with health checks, metrics, and proper error handling.
"""

import os
import json
import logging
from typing import List, Dict, Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Wine Classifier API",
    description="ML-powered Wine Cultivar Classification Service - Deployed on Kubernetes",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Global variables for model and metadata
model = None
metadata = None
feature_names = None
target_names = None

class WineFeatures(BaseModel):
    """Input features for wine classification (must match sklearn wine dataset order)"""
    alcohol: float = Field(..., description="Alcohol content", example=13.0)
    malic_acid: float = Field(..., description="Malic acid", example=2.0)
    ash: float = Field(..., description="Ash", example=2.5)
    alcalinity_of_ash: float = Field(..., description="Alcalinity of ash", example=15.0)
    magnesium: float = Field(..., description="Magnesium", example=100.0)
    total_phenols: float = Field(..., description="Total phenols", example=2.5)
    flavanoids: float = Field(..., description="Flavanoids", example=2.0)
    nonflavanoid_phenols: float = Field(..., description="Nonflavanoid phenols", example=0.3)
    proanthocyanins: float = Field(..., description="Proanthocyanins", example=1.5)
    color_intensity: float = Field(..., description="Color intensity", example=5.0)
    hue: float = Field(..., description="Hue", example=1.0)
    od280_od315_of_diluted_wines: float = Field(..., description="OD280/OD315 of diluted wines", example=3.0)
    proline: float = Field(..., description="Proline", example=800.0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "alcohol": 13.72,
                "malic_acid": 1.43,
                "ash": 2.50,
                "alcalinity_of_ash": 16.7,
                "magnesium": 108.0,
                "total_phenols": 3.40,
                "flavanoids": 3.67,
                "nonflavanoid_phenols": 0.19,
                "proanthocyanins": 2.04,
                "color_intensity": 6.80,
                "hue": 0.89,
                "od280_od315_of_diluted_wines": 2.87,
                "proline": 1285.0
            }
        }
    }

class PredictionResponse(BaseModel):
    """Response model for prediction"""
    predicted_class: int
    predicted_label: str
    confidence: float
    probabilities: Dict[str, float]
    model_version: str = "1.0.0"

@app.on_event("startup")
async def load_model():
    """Load model and metadata on application startup"""
    global model, metadata, feature_names, target_names
    
    model_path = "models/wine_classifier.pkl"
    metadata_path = "models/metadata.json"
    
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}. Please run 'python train.py' first.")
        
        model = joblib.load(model_path)
        logger.info(f"✅ Model loaded successfully from {model_path}")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            feature_names = metadata.get("feature_names", [])
            target_names = metadata.get("target_names", [])
            logger.info(f"✅ Metadata loaded. Features: {len(feature_names)}, Classes: {target_names}")
        else:
            # Fallback if no metadata
            from sklearn.datasets import load_wine
            wine = load_wine()
            feature_names = list(wine.feature_names)
            target_names = list(wine.target_names)
            logger.warning("Metadata not found, using fallback feature names")
            
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        raise

@app.get("/", tags=["General"])
async def root():
    return {
        "message": "Wine Classifier ML API is running!",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "metrics": "/metrics"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Kubernetes liveness/readiness probe endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": metadata.get("model_type", "RandomForestClassifier") if metadata else "Unknown"
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(features: WineFeatures):
    """
    Predict wine cultivar class from chemical features.
    
    Returns predicted class (0, 1, or 2) and probabilities for each class.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please wait.")
    
    try:
        # Convert Pydantic model to numpy array in correct order
        input_data = np.array([[
            features.alcohol,
            features.malic_acid,
            features.ash,
            features.alcalinity_of_ash,
            features.magnesium,
            features.total_phenols,
            features.flavanoids,
            features.nonflavanoid_phenols,
            features.proanthocyanins,
            features.color_intensity,
            features.hue,
            features.od280_od315_of_diluted_wines,
            features.proline
        ]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        # Build response
        result = PredictionResponse(
            predicted_class=int(prediction),
            predicted_label=target_names[prediction] if target_names else f"class_{prediction}",
            confidence=float(np.max(probabilities)),
            probabilities={
                target_names[i] if target_names else f"class_{i}": float(prob) 
                for i, prob in enumerate(probabilities)
            }
        )
        
        logger.info(f"Prediction made: Class {result.predicted_class} ({result.predicted_label}) with confidence {result.confidence:.3f}")
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.get("/model-info", tags=["Model"])
async def model_info():
    """Get information about the loaded model"""
    if metadata is None:
        raise HTTPException(status_code=503, detail="Metadata not available")
    
    return {
        "model_type": metadata.get("model_type"),
        "accuracy": metadata.get("accuracy"),
        "n_features": metadata.get("n_features"),
        "feature_names": feature_names,
        "target_names": target_names,
        "classes": {i: name for i, name in enumerate(target_names)} if target_names else {}
    }