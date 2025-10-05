from fastapi.routing import APIRouter
from fastapi import UploadFile, File, HTTPException, Form
from pathlib import Path
import tempfile
import os
import json
import pandas as pd
import pickle
import numpy as np
import io
from typing import Dict, Any

router = APIRouter()

# Base models directory
MODELS_BASE_DIR = Path("/Users/hamzaboulaala/Documents/github/NASA-hack/models/2025.10.05_15.15.52")


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess uploaded data to match training format"""
    try:
        # Handle missing values
        df = df.fillna(df.median(numeric_only=True))
        
        # Ensure numeric columns are properly typed
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].astype(float)
        
        # Remove any non-numeric columns that might cause issues
        df = df.select_dtypes(include=[np.number])
        
        return df
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data preprocessing failed: {str(e)}")


def get_feature_importance(pipeline, feature_names) -> list:
    """Extract feature importance from the model"""
    try:
        # Get the final estimator from the pipeline
        if hasattr(pipeline, 'named_steps'):
            estimator = pipeline.named_steps.get('model', pipeline)
        else:
            estimator = pipeline
            
        # Extract feature importance
        if hasattr(estimator, 'feature_importances_'):
            importances = estimator.feature_importances_
        elif hasattr(estimator, 'coef_'):
            importances = np.abs(estimator.coef_[0])
        else:
            return []
            
        # Create feature importance list
        feature_imp = [(name, float(imp)) for name, imp in zip(feature_names, importances)]
        feature_imp.sort(key=lambda x: x[1], reverse=True)
        
        return feature_imp
        
    except Exception:
        return []


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    telescope: str = Form(...),
    model_type: str = Form(...)
):
    """
    Make predictions on uploaded data using specified telescope and model type
    
    Parameters:
    - file: CSV or JSON file with exoplanet data
    - telescope: Either "KEPLER" or "TESS"
    - model_type: Model name like "Logistic_Regression", "Decision_Tree", or "Random_Forest"
    """
    
    # Validate inputs
    if telescope.upper() not in ["KEPLER", "TESS"]:
        raise HTTPException(
            status_code=400, 
            detail="Telescope must be either 'KEPLER' or 'TESS'"
        )
    
    if not file.filename.lower().endswith(('.csv', '.json')):
        raise HTTPException(
            status_code=400, 
            detail="Only CSV and JSON files are supported"
        )
    
    try:
        # Read the uploaded file
        content = await file.read()
        
        # Parse based on file type
        if file.filename.lower().endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        elif file.filename.lower().endswith('.json'):
            df = pd.read_json(io.BytesIO(content))
        
        # Validate data
        if df.empty:
            raise HTTPException(status_code=400, detail="Empty dataset")
        
        # Construct model path
        telescope_upper = telescope.upper()
        model_filename = f"{model_type}_model.pkl"
        model_path = MODELS_BASE_DIR / telescope_upper / model_filename
        
        if not model_path.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"Model not found: {model_filename} for {telescope_upper}"
            )
        
        # Load the model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Extract the trained pipeline
        if isinstance(model_data, dict) and 'trained_pipeline' in model_data:
            trained_pipeline = model_data['trained_pipeline']
        else:
            trained_pipeline = model_data
        
        # Preprocess the data
        processed_data = preprocess_data(df)
        
        # Make predictions
        predictions = trained_pipeline.predict(processed_data)
        probabilities = trained_pipeline.predict_proba(processed_data)
        
        # Prepare results for all rows
        results = []
        for idx in range(len(predictions)):
            is_exoplanet = predictions[idx] == 1
            confidence = float(np.max(probabilities[idx]))
            
            result = {
                "row": idx,
                "prediction": "exoplanet" if is_exoplanet else "not_exoplanet",
                "confidence": confidence,
                "probabilities": {
                    "exoplanet": float(probabilities[idx][1]),
                    "not_exoplanet": float(probabilities[idx][0])
                }
            }
            results.append(result)
        
        # Get feature importance
        # feature_importance = get_feature_importance(trained_pipeline, processed_data.columns)
        
        # Prepare final response
        response = {
            "telescope": telescope_upper,
            "model": model_type,
            "total_predictions": len(predictions),
            "summary": {
                "exoplanets_detected": int(np.sum(predictions == 1)),
                "non_exoplanets": int(np.sum(predictions == 0)),
                "average_confidence": float(np.mean([r["confidence"] for r in results]))
            },
            "predictions": results,
            "analysis": {
                # "top_features": feature_importance[:10],  # Top 10 features
                "feature_count": len(processed_data.columns),
                "rows_processed": len(processed_data)
            }
        }
        
        return response
        
    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=400, 
            detail="CSV file is empty or invalid"
        )
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400, 
            detail="Invalid JSON format"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing file: {str(e)}"
        )