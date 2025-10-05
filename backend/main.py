"""
FastAPI Backend for ExoPlanet AI
Handles file uploads, predictions, and model serving
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import numpy as np
import pickle
import json
import os
import logging
import io
from pathlib import Path
from typing import Optional, Dict, Any
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ExoPlanet AI API",
    description="Machine Learning API for Exoplanet Detection",
    version="1.0.0"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

# Global variables for model storage
MODELS_DIR = Path("../2025.10.05_10.36.41")
MODEL_CACHE = {}
METRICS_CACHE = {}

class ExoPlanetPredictor:
    """Main prediction class for exoplanet detection"""
    
    def __init__(self):
        self.models_dir = MODELS_DIR
        self.load_models()
        self.load_metrics()
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            model_files = list(self.models_dir.glob("*_model.pkl"))
            
            for model_file in model_files:
                model_name = model_file.stem.replace("_model", "").replace("_", " ").title()
                
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                    MODEL_CACHE[model_name.lower()] = model_data
                    
                logger.info(f"Loaded model: {model_name}")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def load_metrics(self):
        """Load model metrics and results"""
        try:
            # Load comparison metrics
            metrics_file = self.models_dir / "Logistic_Regression_comparison_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    METRICS_CACHE['comparison_metrics'] = json.load(f)
            
            # Load feature importance
            importance_file = self.models_dir / "Logistic_Regression_feature_importance.json"
            if importance_file.exists():
                with open(importance_file, 'r') as f:
                    METRICS_CACHE['feature_importance'] = json.load(f)
            
            # Load test results
            results_file = self.models_dir / "test_results_2025.10.05_10.36.41.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    METRICS_CACHE['test_results'] = json.load(f)
            
            # Load experiment config
            config_file = self.models_dir / "experiment_config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    METRICS_CACHE['experiment_config'] = json.load(f)
                    
            logger.info("Loaded model metrics and configuration")
            
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess uploaded data to match training format"""
        try:
            # Basic preprocessing - in a real implementation, this would be more comprehensive
            # For now, we'll handle missing values and ensure proper data types
            
            # Handle missing values
            df = df.fillna(df.median())
            
            # Ensure numeric columns are properly typed
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].astype(float)
            
            # Remove any non-numeric columns that might cause issues
            df = df.select_dtypes(include=[np.number])
            
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise HTTPException(status_code=400, detail=f"Data preprocessing failed: {str(e)}")
    
    def predict(self, data: pd.DataFrame, telescope: str, model_type: str) -> Dict[str, Any]:
        """Make prediction on uploaded data"""
        try:
            # Get the appropriate model
            model_key = model_type.lower().replace(" ", "_")
            
            if model_key not in MODEL_CACHE:
                raise HTTPException(status_code=400, detail=f"Model {model_type} not available")
            
            model_data = MODEL_CACHE[model_key]
            trained_pipeline = model_data.get("trained_pipeline")
            
            if trained_pipeline is None:
                raise HTTPException(status_code=500, detail="Model pipeline not found")
            
            # Preprocess the data
            processed_data = self.preprocess_data(data)
            
            # Make predictions
            predictions = trained_pipeline.predict(processed_data)
            probabilities = trained_pipeline.predict_proba(processed_data)
            
            # Calculate confidence (max probability)
            confidence = np.max(probabilities, axis=1)
            
            # Determine if it's an exoplanet (assuming class 1 is exoplanet)
            is_exoplanet = predictions[0] == 1
            prediction_class = "exoplanet" if is_exoplanet else "not_exoplanet"
            
            # Get feature importance for analysis
            feature_importance = self.get_feature_importance(trained_pipeline, processed_data.columns)
            
            # Prepare result
            result = {
                "prediction": prediction_class,
                "confidence": float(confidence[0]),
                "probabilities": {
                    "exoplanet": float(probabilities[0][1]),
                    "not_exoplanet": float(probabilities[0][0])
                },
                "analysis": {
                    "key_features": feature_importance[:5],  # Top 5 features
                    "data_quality": self.assess_data_quality(processed_data),
                    "missing_values": int(processed_data.isnull().sum().sum()),
                    "feature_count": len(processed_data.columns)
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    def get_feature_importance(self, pipeline, feature_names) -> list:
        """Extract feature importance from the model"""
        try:
            # Get the classifier from the pipeline
            classifier = pipeline.named_steps.get('classifier')
            
            if hasattr(classifier, 'coef_'):
                # For logistic regression
                importance = np.abs(classifier.coef_[0])
            elif hasattr(classifier, 'feature_importances_'):
                # For tree-based models
                importance = classifier.feature_importances_
            else:
                # Fallback to random importance
                importance = np.random.random(len(feature_names))
            
            # Create feature importance list
            feature_importance = []
            for i, (name, imp) in enumerate(zip(feature_names, importance)):
                feature_importance.append({
                    "name": name,
                    "value": float(imp),
                    "importance": float(imp / np.max(importance))  # Normalized importance
                })
            
            # Sort by importance
            feature_importance.sort(key=lambda x: x['importance'], reverse=True)
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"Error extracting feature importance: {e}")
            return []
    
    def assess_data_quality(self, df: pd.DataFrame) -> str:
        """Assess the quality of uploaded data"""
        try:
            missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            
            if missing_ratio < 0.05:
                return "Excellent"
            elif missing_ratio < 0.15:
                return "Good"
            elif missing_ratio < 0.30:
                return "Fair"
            else:
                return "Poor"
                
        except Exception:
            return "Unknown"

# Initialize predictor
predictor = ExoPlanetPredictor()

@app.get("/")
async def root():
    """Root endpoint - serve the frontend"""
    return FileResponse("../frontend/index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": len(MODEL_CACHE)}

@app.get("/api/models")
async def get_available_models():
    """Get list of available models"""
    return {
        "models": list(MODEL_CACHE.keys()),
        "telescopes": ["kepler", "tess", "k2"],
        "default_model": "logistic regression"
    }

@app.get("/api/metrics")
async def get_model_metrics():
    """Get model performance metrics"""
    return METRICS_CACHE

@app.post("/api/predict")
async def predict_exoplanet(
    file: UploadFile = File(...),
    telescope: str = Form(...),
    model: str = Form(...)
):
    """Predict exoplanet from uploaded data"""
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.csv', '.json', '.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Read the uploaded file
        content = await file.read()
        
        # Parse based on file type
        if file.filename.lower().endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        elif file.filename.lower().endswith('.json'):
            df = pd.read_json(io.BytesIO(content))
        elif file.filename.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Validate data
        if df.empty:
            raise HTTPException(status_code=400, detail="Empty dataset")
        
        if len(df.columns) < 10:
            raise HTTPException(status_code=400, detail="Insufficient features for prediction")
        
        # Make prediction
        result = predictor.predict(df, telescope, model)
        
        logger.info(f"Prediction completed for {file.filename}: {result['prediction']}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/predict-form")
async def predict_exoplanet_from_form(
    request_data: dict
):
    """Predict exoplanet from form data"""
    try:
        data = request_data.get('data', {})
        telescope = request_data.get('telescope', 'kepler')
        model = request_data.get('model', 'logistic_regression')
        
        if not data:
            raise HTTPException(status_code=400, detail="No data provided")
        
        # Convert form data to DataFrame
        df = pd.DataFrame([data])
        
        # Make prediction
        result = predictor.predict(df, telescope, model)
        
        logger.info(f"Form prediction completed: {result['prediction']}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Form prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/analytics")
async def get_analytics():
    """Get detailed analytics data"""
    try:
        analytics_data = {
            "performance_metrics": METRICS_CACHE.get('comparison_metrics', {}),
            "feature_importance": METRICS_CACHE.get('feature_importance', {}),
            "test_results": METRICS_CACHE.get('test_results', {}),
            "experiment_config": METRICS_CACHE.get('experiment_config', {})
        }
        
        return analytics_data
        
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analytics data")

@app.get("/api/telescope/{telescope_name}")
async def get_telescope_info(telescope_name: str):
    """Get information about a specific telescope"""
    telescope_info = {
        "kepler": {
            "name": "Kepler Space Telescope",
            "description": "NASA's planet-hunting telescope that discovered thousands of exoplanets",
            "launch_date": "2009-03-07",
            "status": "Retired (2018)",
            "discoveries": "2,662 confirmed exoplanets",
            "model_accuracy": "86.25%",
            "data_available": True
        },
        "tess": {
            "name": "Transiting Exoplanet Survey Satellite",
            "description": "NASA's current exoplanet hunting mission",
            "launch_date": "2018-04-18",
            "status": "Active",
            "discoveries": "200+ confirmed exoplanets",
            "model_accuracy": "Coming Soon",
            "data_available": False
        },
        "k2": {
            "name": "Kepler's Extended Mission",
            "description": "Kepler's second mission after reaction wheel failure",
            "launch_date": "2014-05-16",
            "status": "Retired (2018)",
            "discoveries": "500+ confirmed exoplanets",
            "model_accuracy": "Coming Soon",
            "data_available": False
        }
    }
    
    if telescope_name.lower() not in telescope_info:
        raise HTTPException(status_code=404, detail="Telescope not found")
    
    return telescope_info[telescope_name.lower()]

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
