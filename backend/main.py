s"""
FastAPI Backend for ExoPlanet AI
Handles file uploads, predictions, and model serving
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
import pickle
import json
import logging
import io
from pathlib import Path
from typing import Dict, Any
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
# Mount individual directories at root level for direct access
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/styles", StaticFiles(directory=str(FRONTEND_DIR / "styles")), name="styles")
app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Global variables for model storage
MODELS_DIR = Path("/Users/hamzaboulaala/Documents/github/NASA-hack/models/2025.10.05_16.07.33")
MODEL_CACHE = {}
METRICS_CACHE = {
    'KEPLER': {},
    'TESS': {}
}

# Mount the models directory to serve static files like confusion matrix images
MODELS_ROOT = Path(__file__).parent.parent / "models"
app.mount("/models", StaticFiles(directory=str(MODELS_ROOT)), name="models")


class ExoPlanetPredictor:
    """Main prediction class for exoplanet detection"""
    
    def __init__(self):
        self.models_dir = MODELS_DIR
        self.load_models()
        self.load_metrics()
    
    def load_models(self):
        """Load trained models from both KEPLER and TESS folders"""
        try:
            # List of available models
            model_names = ['XGBoost', 'Gradient_Boosting', 'Random_Forest', 'Decision_Tree', 'Logistic_Regression']
            
            for telescope in ['KEPLER', 'TESS']:
                telescope_dir = self.models_dir / telescope
                if not telescope_dir.exists():
                    logger.warning(f"Directory not found: {telescope_dir}")
                    continue
                
                logger.info(f"Loading models from: {telescope_dir}")
                
                for model_name in model_names:
                    model_file = telescope_dir / f"{model_name}_model.pkl"
                    
                    if model_file.exists():
                        try:
                            with open(model_file, 'rb') as f:
                                model_data = pickle.load(f)
                                
                            # Create cache key: lowercase telescope + lowercase model with underscores
                            cache_key = f"{telescope.lower()}_{model_name.lower()}"
                            MODEL_CACHE[cache_key] = model_data
                            
                            logger.info(f"✓ Loaded: {telescope}/{model_name} → cache_key: {cache_key}")
                        except Exception as e:
                            logger.error(f"✗ Failed to load {telescope}/{model_name}: {e}")
                    else:
                        logger.warning(f"✗ Model file not found: {model_file}")
            
            logger.info(f"Total models loaded: {len(MODEL_CACHE)}")
            logger.info(f"Available model keys: {list(MODEL_CACHE.keys())}")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def load_metrics(self):
        """Load model metrics and results for both telescopes"""
        try:
            for telescope in ['KEPLER', 'TESS']:
                telescope_dir = self.models_dir / telescope
                if not telescope_dir.exists():
                    logger.warning(f"Directory not found: {telescope_dir}")
                    continue
                
                # Load all model comparison metrics
                model_files = list(telescope_dir.glob("*_comparison_metrics.json"))
                METRICS_CACHE[telescope]['model_metrics'] = {}
                
                for metrics_file in model_files:
                    model_name = metrics_file.stem.replace("_comparison_metrics", "")
                    with open(metrics_file, 'r') as f:
                        METRICS_CACHE[telescope]['model_metrics'][model_name] = json.load(f)
                    logger.info(f"Loaded comparison metrics for {telescope} - {model_name}")
                
                # Load test results with all models
                test_results_files = list(telescope_dir.glob("test_results_*.json"))
                if test_results_files:
                    with open(test_results_files[0], 'r') as f:
                        METRICS_CACHE[telescope]['test_results'] = json.load(f)
                
                # Load experiment config
                config_file = telescope_dir / "experiment_config.json"
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        METRICS_CACHE[telescope]['experiment_config'] = json.load(f)
                
                # Load CV results for all models
                cv_files = list(telescope_dir.glob("*_cv_results.json"))
                METRICS_CACHE[telescope]['cv_results'] = {}
                
                for cv_file in cv_files:
                    model_name = cv_file.stem.replace("_cv_results", "")
                    with open(cv_file, 'r') as f:
                        METRICS_CACHE[telescope]['cv_results'][model_name] = json.load(f)
                        
                logger.info(f"Loaded metrics for {telescope}")
            
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
    
    def preprocess_data(self, df: pd.DataFrame, trained_pipeline=None) -> pd.DataFrame:
        """Preprocess uploaded data to match training format"""
        try:
            # Remove non-feature columns that are typically present in raw data
            columns_to_drop = [
                'rowid', 'kepid', 'kepoi_name', 'kepler_name', 
                'koi_disposition', 'koi_pdisposition', 'koi_vet_stat', 
                'koi_vet_date', 'koi_comment', 'koi_disp_prov',
                'koi_tce_delivname', 'koi_datalink_dvr', 'koi_datalink_dvs',
                'koi_parm_prov', 'koi_sparprov', 'koi_limbdark_mod', 'koi_trans_mod',
                'koi_fittype'
            ]
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
            
            # Select only numeric columns
            df = df.select_dtypes(include=[np.number])
            
            # Ensure proper data types
            df = df.astype(float)
            
            # If pipeline is provided, try to get expected features
            if trained_pipeline and hasattr(trained_pipeline, 'feature_names_in_'):
                expected_features = trained_pipeline.feature_names_in_
                # Add missing features with zeros
                for feature in expected_features:
                    if feature not in df.columns:
                        df[feature] = np.nan
                # Keep only expected features in the right order
                df = df[expected_features]
                # Ensure all columns are float after feature alignment
                df = df.astype(float)
            
            logger.info(f"Preprocessed data shape: {df.shape}, columns: {len(df.columns)}")
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise HTTPException(status_code=400, detail=f"Data preprocessing failed: {str(e)}")
    
    def predict(self, data: pd.DataFrame, telescope: str, model_type: str) -> Dict[str, Any]:
        """Make prediction on uploaded data"""
        try:
            # Normalize model name: replace spaces with underscores and convert to lowercase
            # This handles inputs like "XGBoost", "Gradient Boosting", "gradient_boosting", "xgboost"
            normalized_model = model_type.lower().replace(' ', '_')
            model_key = f"{telescope.lower()}_{normalized_model}"
            
            logger.info(f"Looking for model: {model_key}")
            logger.info(f"Available models: {list(MODEL_CACHE.keys())}")
            
            if model_key not in MODEL_CACHE:
                available_models = [k for k in MODEL_CACHE.keys() if k.startswith(telescope.lower())]
                logger.error(f"Model key '{model_key}' not found. Available models: {available_models}")
                raise HTTPException(
                    status_code=400, 
                    detail=f"Model {model_type} for {telescope} not available. Available: {available_models}"
                )
            
            model_data = MODEL_CACHE[model_key]
            trained_pipeline = model_data.get("trained_pipeline")
            
            if trained_pipeline is None:
                raise HTTPException(status_code=500, detail="Model pipeline not found")
            
            # Preprocess the data with pipeline feature alignment
            processed_data = self.preprocess_data(data, trained_pipeline)
            
            logger.info(f"Making prediction with {model_key} on {len(processed_data)} samples")
            
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
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/demo_data.json")
async def get_demo_data():
    """Serve demo data for the frontend"""
    demo_file = FRONTEND_DIR / "demo_data.json"
    if not demo_file.exists():
        raise HTTPException(status_code=404, detail="Demo data not found")
    return FileResponse(str(demo_file))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": len(MODEL_CACHE)}

@app.get("/api/models")
async def get_available_models():
    """Get list of available models"""
    # Group models by telescope
    kepler_models = []
    tess_models = []
    
    for key in MODEL_CACHE.keys():
        # key format is "telescope_model_name", e.g., "kepler_xgboost", "kepler_gradient_boosting"
        parts = key.split('_', 1)
        if len(parts) > 1:
            telescope = parts[0]
            model_name_key = parts[1]  # e.g., "xgboost", "gradient_boosting"
            
            # Convert to display name (keep underscores, just capitalize)
            # "xgboost" -> "XGBoost", "gradient_boosting" -> "Gradient_Boosting"
            if model_name_key == 'xgboost':
                display_name = 'XGBoost'
            elif model_name_key == 'gradient_boosting':
                display_name = 'Gradient_Boosting'
            elif model_name_key == 'random_forest':
                display_name = 'Random_Forest'
            elif model_name_key == 'decision_tree':
                display_name = 'Decision_Tree'
            elif model_name_key == 'logistic_regression':
                display_name = 'Logistic_Regression'
            else:
                display_name = model_name_key.replace('_', ' ').title()
            
            if telescope == 'kepler':
                kepler_models.append(display_name)
            elif telescope == 'tess':
                tess_models.append(display_name)
    
    return {
        "models": list(MODEL_CACHE.keys()),
        "kepler_models": sorted(kepler_models),
        "tess_models": sorted(tess_models),
        "telescopes": ["kepler", "tess"],
        "default_model": "XGBoost",
        "default_telescope": "kepler",
        "total_models": len(MODEL_CACHE)
    }

@app.get("/api/metrics")
async def get_model_metrics(telescope: str = "KEPLER"):
    """Get model performance metrics for a specific telescope"""
    telescope_upper = telescope.upper()
    if telescope_upper not in METRICS_CACHE:
        raise HTTPException(status_code=404, detail=f"Metrics for {telescope} not found")
    return {
        "telescope": telescope_upper,
        "metrics": METRICS_CACHE[telescope_upper]
    }

@app.get("/api/metrics/all")
async def get_all_metrics():
    """Get model performance metrics for all telescopes"""
    return METRICS_CACHE

@app.post("/api/predict")
async def predict_exoplanet(
    file: UploadFile = File(...),
    telescope: str = Form(default="kepler"),
    model: str = Form(default="XGBoost")
):
    """Predict exoplanet from uploaded data"""
    try:
        logger.info(f"Received prediction request: file={file.filename}, telescope={telescope}, model={model}")
        
        # Validate file type
        if not file.filename.lower().endswith(('.csv', '.json', '.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload CSV, JSON, XLSX, or XLS file.")
        
        # Read the uploaded file
        content = await file.read()
        
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Parse based on file type
        try:
            if file.filename.lower().endswith('.csv'):
                df = pd.read_csv(io.BytesIO(content))
            elif file.filename.lower().endswith('.json'):
                df = pd.read_json(io.BytesIO(content))
            elif file.filename.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(io.BytesIO(content))
            else:
                raise HTTPException(status_code=400, detail="Unsupported file format")
        except Exception as parse_error:
            logger.error(f"File parsing error: {parse_error}")
            raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(parse_error)}")
        
        # Validate data
        if df.empty:
            raise HTTPException(status_code=400, detail="Empty dataset - no rows found in file")
        
        if len(df.columns) < 5:
            raise HTTPException(status_code=400, detail=f"Insufficient features for prediction. Found {len(df.columns)} columns, need at least 5.")
        
        logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Make prediction
        result = predictor.predict(df, telescope, model)
        
        logger.info(f"Prediction completed for {file.filename}: {result['prediction']} with {result['confidence']:.2%} confidence")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
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
async def get_analytics(telescope: str = "KEPLER", model: str = None):
    """Get detailed analytics data for a specific telescope and optionally a specific model"""
    try:
        telescope_upper = telescope.upper()
        if telescope_upper not in METRICS_CACHE:
            raise HTTPException(status_code=404, detail=f"Analytics for {telescope} not found")
        
        telescope_data = METRICS_CACHE[telescope_upper]
        
        # If a specific model is requested
        if model:
            model_name = model.replace(" ", "_")
            model_metrics = telescope_data.get('model_metrics', {}).get(model_name, {})
            cv_results = telescope_data.get('cv_results', {}).get(model_name, {})
            
            analytics_data = {
                "telescope": telescope_upper,
                "model": model,
                "performance_metrics": model_metrics,
                "cv_results": cv_results,
                "experiment_config": telescope_data.get('experiment_config', {})
            }
        else:
            # Return all models data
            analytics_data = {
                "telescope": telescope_upper,
                "model_metrics": telescope_data.get('model_metrics', {}),
                "test_results": telescope_data.get('test_results', {}),
                "experiment_config": telescope_data.get('experiment_config', {}),
                "cv_results": telescope_data.get('cv_results', {})
            }
        
        return analytics_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analytics data")

@app.get("/api/telescope/{telescope_name}")
async def get_telescope_info(telescope_name: str):
    """Get information about a specific telescope with actual model metrics"""
    # Get actual metrics if available
    kepler_accuracy = "N/A"
    tess_accuracy = "N/A"
    kepler_samples = "N/A"
    tess_samples = "N/A"
    kepler_features = "N/A"
    tess_features = "N/A"
    
    # Get best model accuracy for KEPLER (using XGBoost which has highest accuracy)
    if 'KEPLER' in METRICS_CACHE and 'test_results' in METRICS_CACHE['KEPLER']:
        test_results = METRICS_CACHE['KEPLER']['test_results']
        if 'metrics' in test_results and 'XGBoost' in test_results['metrics']:
            xgb_metrics = test_results['metrics']['XGBoost']
            if 'values' in xgb_metrics and len(xgb_metrics['values']) > 0:
                kepler_accuracy = f"{xgb_metrics['values'][0] * 100:.2f}"
    
    if 'KEPLER' in METRICS_CACHE and 'experiment_config' in METRICS_CACHE['KEPLER']:
        config = METRICS_CACHE['KEPLER']['experiment_config']
        kepler_samples = str(config.get('n_train', 'N/A'))
        kepler_features = str(config.get('n_features', 'N/A'))
    
    # Get metrics for TESS if available
    if 'TESS' in METRICS_CACHE and 'test_results' in METRICS_CACHE['TESS']:
        test_results = METRICS_CACHE['TESS']['test_results']
        if 'metrics' in test_results:
            # Get first available model's accuracy
            for model_name, model_metrics in test_results['metrics'].items():
                if 'values' in model_metrics and len(model_metrics['values']) > 0:
                    tess_accuracy = f"{model_metrics['values'][0] * 100:.2f}"
                    break
    
    if 'TESS' in METRICS_CACHE and 'experiment_config' in METRICS_CACHE['TESS']:
        config = METRICS_CACHE['TESS']['experiment_config']
        tess_samples = str(config.get('n_train', 'N/A'))
        tess_features = str(config.get('n_features', 'N/A'))
    
    telescope_info = {
        "kepler": {
            "name": "Kepler Space Telescope",
            "description": "NASA's planet-hunting telescope that discovered thousands of exoplanets",
            "launch_date": "2009-03-07",
            "status": "Retired (2018)",
            "discoveries": "2,662 confirmed exoplanets",
            "model_accuracy": kepler_accuracy,
            "training_samples": kepler_samples,
            "features": kepler_features,
            "data_available": True
        },
        "tess": {
            "name": "Transiting Exoplanet Survey Satellite",
            "description": "NASA's current exoplanet hunting mission",
            "launch_date": "2018-04-18",
            "status": "Active",
            "discoveries": "200+ confirmed exoplanets",
            "model_accuracy": tess_accuracy,
            "training_samples": tess_samples,
            "features": tess_features,
            "data_available": True
        },
        "k2": {
            "name": "Kepler's Extended Mission",
            "description": "Kepler's second mission after reaction wheel failure",
            "launch_date": "2014-05-16",
            "status": "Retired (2018)",
            "discoveries": "500+ confirmed exoplanets",
            "model_accuracy": "Coming Soon",
            "training_samples": "N/A",
            "features": "N/A",
            "data_available": False
        }
    }
    
    if telescope_name.lower() not in telescope_info:
        raise HTTPException(status_code=404, detail="Telescope not found")
    
    return telescope_info[telescope_name.lower()]

@app.get("/api/confusion_matrix/{telescope}")
async def get_confusion_matrix(telescope: str, model: str = "XGBoost"):
    """Get the path to confusion matrix image for a specific telescope and model"""
    telescope_upper = telescope.upper()
    if telescope_upper not in ['KEPLER', 'TESS']:
        raise HTTPException(status_code=404, detail=f"Telescope {telescope} not found")
    
    # Normalize model name (replace spaces with underscores)
    model_name = model.replace(" ", "_")
    
    # Construct the image path
    experiment_id = MODELS_DIR.name
    image_path = f"/models/{experiment_id}/{telescope_upper}/{model_name}_confusion_matrix.png"
    
    # Check if the file exists
    actual_path = MODELS_DIR / telescope_upper / f"{model_name}_confusion_matrix.png"
    if not actual_path.exists():
        raise HTTPException(status_code=404, detail=f"Confusion matrix not found for {telescope} - {model}")
    
    return {
        "telescope": telescope_upper,
        "model": model,
        "image_url": image_path,
        "experiment_id": experiment_id
    }

@app.get("/api/feature_importance/{telescope}")
async def get_feature_importance(telescope: str, model: str = "XGBoost"):
    """Get feature importance data for a specific telescope and model"""
    telescope_upper = telescope.upper()
    if telescope_upper not in ['KEPLER', 'TESS']:
        raise HTTPException(status_code=404, detail=f"Telescope {telescope} not found")
    
    telescope_dir = MODELS_DIR / telescope_upper
    model_name = model.replace(" ", "_")
    
    # Extract from model file
    model_file = telescope_dir / f"{model_name}_model.pkl"
    if model_file.exists():
        try:
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
                trained_pipeline = model_data.get("trained_pipeline")
                
                if trained_pipeline and hasattr(trained_pipeline, 'named_steps'):
                    classifier = trained_pipeline.named_steps.get('classifier')
                    
                    # Get feature names from preprocessor if available
                    preprocessor = trained_pipeline.named_steps.get('preprocessor')
                    feature_names = None
                    
                    if preprocessor and hasattr(preprocessor, 'get_feature_names_out'):
                        try:
                            feature_names = list(preprocessor.get_feature_names_out())
                        except Exception:
                            pass
                    
                    # Get feature importance/coefficients
                    importance_values = None
                    if hasattr(classifier, 'coef_'):
                        # For logistic regression
                        importance_values = np.abs(classifier.coef_[0]).tolist()
                    elif hasattr(classifier, 'feature_importances_'):
                        # For tree-based models (XGBoost, Random Forest, etc.)
                        importance_values = classifier.feature_importances_.tolist()
                    
                    if importance_values and feature_names and len(importance_values) == len(feature_names):
                        # Sort by importance
                        feature_importance_pairs = list(zip(feature_names, importance_values))
                        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
                        
                        # Take top 15
                        top_features = feature_importance_pairs[:15]
                        
                        return {
                            "telescope": telescope_upper,
                            "model": model,
                            "features": [name for name, _ in top_features],
                            "importance_values": [val for _, val in top_features],
                            "n_features": len(feature_names),
                            "source": "model_coefficients"
                        }
        except Exception as e:
            logger.error(f"Error extracting feature importance: {e}")
    
    raise HTTPException(status_code=404, detail=f"Feature importance data not found for {telescope} - {model}")

@app.get("/api/cv_results/{telescope}")
async def get_cv_results(telescope: str, model: str = "XGBoost"):
    """Get cross-validation results for a specific telescope and model"""
    telescope_upper = telescope.upper()
    if telescope_upper not in ['KEPLER', 'TESS']:
        raise HTTPException(status_code=404, detail=f"Telescope {telescope} not found")
    
    telescope_dir = MODELS_DIR / telescope_upper
    model_name = model.replace(" ", "_")
    
    # Try to load CV results from JSON file
    cv_results_file = telescope_dir / f"{model_name}_cv_results.json"
    
    if cv_results_file.exists():
        try:
            with open(cv_results_file, 'r') as f:
                cv_data = json.load(f)
                
                return {
                    "telescope": telescope_upper,
                    "model": model,
                    "best_score": cv_data.get("best_score"),
                    "best_params": cv_data.get("best_params"),
                    "cv_results": {
                        "mean_test_score": cv_data.get("cv_results", {}).get("mean_test_score", []),
                        "std_test_score": cv_data.get("cv_results", {}).get("std_test_score", []),
                        "mean_fit_time": cv_data.get("cv_results", {}).get("mean_fit_time", [])
                    },
                    "source": "cv_results_file"
                }
        except Exception as e:
            logger.error(f"Error loading CV results: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to load CV results: {str(e)}")
    
    raise HTTPException(status_code=404, detail=f"CV results not found for {telescope} - {model}")

@app.get("/api/training_logs/{telescope}")
async def get_training_logs(telescope: str):
    """Get training logs for a specific telescope"""
    telescope_upper = telescope.upper()
    if telescope_upper not in ['KEPLER', 'TESS']:
        raise HTTPException(status_code=404, detail=f"Telescope {telescope} not found")
    
    telescope_dir = MODELS_DIR / telescope_upper
    
    # Find the training log file (pattern: training_*.log)
    log_files = list(telescope_dir.glob("training_*.log"))
    
    if not log_files:
        raise HTTPException(status_code=404, detail=f"Training logs not found for {telescope}")
    
    # Use the first (most recent) log file
    log_file = log_files[0]
    
    try:
        with open(log_file, 'r') as f:
            logs = f.read()
            
        return {
            "telescope": telescope_upper,
            "log_file": log_file.name,
            "logs": logs,
            "line_count": len(logs.split('\n'))
        }
    except Exception as e:
        logger.error(f"Error reading training logs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read training logs: {str(e)}")

@app.get("/api/dashboard/stats")
async def get_dashboard_stats():
    """Get all dashboard statistics dynamically from model files"""
    try:
        stats = {
            "kepler": {
                "accuracy": "N/A",
                "training_samples": "N/A",
                "test_samples": "N/A",
                "features": "N/A",
                "status": "not_ready",
                "models": []
            },
            "tess": {
                "accuracy": "N/A",
                "training_samples": "N/A",
                "test_samples": "N/A",
                "features": "N/A",
                "status": "not_ready",
                "models": []
            }
        }
        
        # Load KEPLER stats
        if 'KEPLER' in METRICS_CACHE:
            kepler_data = METRICS_CACHE['KEPLER']
            
            # Get best accuracy from test results (XGBoost)
            if 'test_results' in kepler_data:
                test_results = kepler_data['test_results']
                if 'metrics' in test_results and 'XGBoost' in test_results['metrics']:
                    xgb_metrics = test_results['metrics']['XGBoost']
                    if 'values' in xgb_metrics and len(xgb_metrics['values']) > 0:
                        stats['kepler']['accuracy'] = f"{xgb_metrics['values'][0] * 100:.2f}"
                        stats['kepler']['status'] = "ready"
                
                # Get all available models
                if 'metrics' in test_results:
                    stats['kepler']['models'] = list(test_results['metrics'].keys())
            
            # Get training/test samples and features from config
            if 'experiment_config' in kepler_data:
                config = kepler_data['experiment_config']
                stats['kepler']['training_samples'] = str(config.get('n_train', 'N/A'))
                stats['kepler']['test_samples'] = str(config.get('n_test', 'N/A'))
                stats['kepler']['features'] = str(config.get('n_features', 'N/A'))
        
        # Load TESS stats
        if 'TESS' in METRICS_CACHE:
            tess_data = METRICS_CACHE['TESS']
            
            # Get accuracy from test results (first available model)
            if 'test_results' in tess_data:
                test_results = tess_data['test_results']
                if 'metrics' in test_results:
                    stats['tess']['models'] = list(test_results['metrics'].keys())
                    # Get first available model's accuracy
                    for model_name, model_metrics in test_results['metrics'].items():
                        if 'values' in model_metrics and len(model_metrics['values']) > 0:
                            stats['tess']['accuracy'] = f"{model_metrics['values'][0] * 100:.2f}"
                            stats['tess']['status'] = "ready"
                            break
            
            # Get training/test samples and features from config
            if 'experiment_config' in tess_data:
                config = tess_data['experiment_config']
                stats['tess']['training_samples'] = str(config.get('n_train', 'N/A'))
                stats['tess']['test_samples'] = str(config.get('n_test', 'N/A'))
                stats['tess']['features'] = str(config.get('n_features', 'N/A'))
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting dashboard stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve dashboard statistics")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
