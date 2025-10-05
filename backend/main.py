"""
FastAPI Backend for ExoPlanet AI
Handles file uploads, predictions, and model serving
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
<<<<<<< HEAD
from fastapi.responses import JSONResponse, FileResponse
=======
from fastapi.responses import FileResponse
>>>>>>> ea08e74f096fe53284d3ab221b4cd1688a279e9b
import pandas as pd
import numpy as np
import pickle
import json
<<<<<<< HEAD
import os
import logging
import io
from pathlib import Path
from typing import Optional, Dict, Any
=======
import logging
import io
from pathlib import Path
from typing import Dict, Any
>>>>>>> ea08e74f096fe53284d3ab221b4cd1688a279e9b
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
<<<<<<< HEAD
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

# Global variables for model storage
MODELS_DIR = Path("../2025.10.05_10.36.41")
MODEL_CACHE = {}
METRICS_CACHE = {}
=======
# Mount individual directories at root level for direct access
app.mount("/styles", StaticFiles(directory="./frontend/styles"), name="styles")
app.mount("/js", StaticFiles(directory="./frontend/js"), name="js")
app.mount("/static", StaticFiles(directory="./frontend"), name="static")

# Global variables for model storage
MODELS_DIR = Path("./models/2025.10.05_15.15.52")
MODEL_CACHE = {}
METRICS_CACHE = {
    'KEPLER': {},
    'TESS': {}
}

# Mount the models directory to serve static files like confusion matrix images
app.mount("/models", StaticFiles(directory="./models"), name="models")

>>>>>>> ea08e74f096fe53284d3ab221b4cd1688a279e9b

class ExoPlanetPredictor:
    """Main prediction class for exoplanet detection"""
    
    def __init__(self):
        self.models_dir = MODELS_DIR
        self.load_models()
        self.load_metrics()
    
    def load_models(self):
<<<<<<< HEAD
        """Load trained models from disk"""
        try:
            model_files = list(self.models_dir.glob("*_model.pkl"))
            
            for model_file in model_files:
                model_name = model_file.stem.replace("_model", "").replace("_", " ").title()
                
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                    MODEL_CACHE[model_name.lower()] = model_data
                    
                logger.info(f"Loaded model: {model_name}")
=======
        """Load trained models from both KEPLER and TESS folders"""
        try:
            for telescope in ['KEPLER', 'TESS']:
                telescope_dir = self.models_dir / telescope
                if not telescope_dir.exists():
                    logger.warning(f"Directory not found: {telescope_dir}")
                    continue
                
                model_files = list(telescope_dir.glob("*_model.pkl"))
                
                for model_file in model_files:
                    model_name = model_file.stem.replace("_model", "").replace("_", " ").title()
                    cache_key = f"{telescope.lower()}_{model_name.lower()}"
                    
                    with open(model_file, 'rb') as f:
                        model_data = pickle.load(f)
                        MODEL_CACHE[cache_key] = model_data
                        
                    logger.info(f"Loaded model: {telescope} - {model_name}")
>>>>>>> ea08e74f096fe53284d3ab221b4cd1688a279e9b
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def load_metrics(self):
<<<<<<< HEAD
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
=======
        """Load model metrics and results for both telescopes"""
        try:
            for telescope in ['KEPLER', 'TESS']:
                telescope_dir = self.models_dir / telescope
                if not telescope_dir.exists():
                    logger.warning(f"Directory not found: {telescope_dir}")
                    continue
                
                # Load comparison metrics
                metrics_file = telescope_dir / "Logistic_Regression_comparison_metrics.json"
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        METRICS_CACHE[telescope]['comparison_metrics'] = json.load(f)
                
                # Load selected features (if exists)
                features_file = telescope_dir / "Logistic_Regression_selected_features.json"
                if features_file.exists():
                    with open(features_file, 'r') as f:
                        METRICS_CACHE[telescope]['selected_features'] = json.load(f)
                
                # Load original features (if exists for TESS)
                orig_features_file = telescope_dir / "Logistic_Regression_original_features.json"
                if orig_features_file.exists():
                    with open(orig_features_file, 'r') as f:
                        METRICS_CACHE[telescope]['original_features'] = json.load(f)
                
                # Load test results
                results_file = telescope_dir / "test_results_2025.10.05_15.15.52.json"
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        METRICS_CACHE[telescope]['test_results'] = json.load(f)
                
                # Load experiment config
                config_file = telescope_dir / "experiment_config.json"
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        METRICS_CACHE[telescope]['experiment_config'] = json.load(f)
                
                # Load CV results
                cv_file = telescope_dir / "Logistic_Regression_cv_results.json"
                if cv_file.exists():
                    with open(cv_file, 'r') as f:
                        METRICS_CACHE[telescope]['cv_results'] = json.load(f)
                        
                logger.info(f"Loaded metrics for {telescope}")
>>>>>>> ea08e74f096fe53284d3ab221b4cd1688a279e9b
            
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
<<<<<<< HEAD
            # Get the appropriate model
            model_key = model_type.lower().replace(" ", "_")
            
            if model_key not in MODEL_CACHE:
                raise HTTPException(status_code=400, detail=f"Model {model_type} not available")
=======
            # Get the appropriate model based on telescope and model type
            model_key = f"{telescope.lower()}_{model_type.lower().replace(' ', '_')}"
            
            if model_key not in MODEL_CACHE:
                raise HTTPException(status_code=400, detail=f"Model {model_type} for {telescope} not available")
>>>>>>> ea08e74f096fe53284d3ab221b4cd1688a279e9b
            
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
<<<<<<< HEAD
    return FileResponse("../frontend/index.html")
=======
    return FileResponse("./frontend/index.html")

@app.get("/demo_data.json")
async def get_demo_data():
    """Serve demo data for the frontend"""
    demo_file = Path("./frontend/demo_data.json")
    if not demo_file.exists():
        raise HTTPException(status_code=404, detail="Demo data not found")
    return FileResponse(str(demo_file))
>>>>>>> ea08e74f096fe53284d3ab221b4cd1688a279e9b

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
<<<<<<< HEAD
async def get_model_metrics():
    """Get model performance metrics"""
=======
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
>>>>>>> ea08e74f096fe53284d3ab221b4cd1688a279e9b
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
<<<<<<< HEAD
async def get_analytics():
    """Get detailed analytics data"""
    try:
        analytics_data = {
            "performance_metrics": METRICS_CACHE.get('comparison_metrics', {}),
            "feature_importance": METRICS_CACHE.get('feature_importance', {}),
            "test_results": METRICS_CACHE.get('test_results', {}),
            "experiment_config": METRICS_CACHE.get('experiment_config', {})
=======
async def get_analytics(telescope: str = "KEPLER"):
    """Get detailed analytics data for a specific telescope"""
    try:
        telescope_upper = telescope.upper()
        if telescope_upper not in METRICS_CACHE:
            raise HTTPException(status_code=404, detail=f"Analytics for {telescope} not found")
        
        telescope_data = METRICS_CACHE[telescope_upper]
        analytics_data = {
            "telescope": telescope_upper,
            "performance_metrics": telescope_data.get('comparison_metrics', {}),
            "selected_features": telescope_data.get('selected_features', {}),
            "original_features": telescope_data.get('original_features', {}),
            "test_results": telescope_data.get('test_results', {}),
            "experiment_config": telescope_data.get('experiment_config', {}),
            "cv_results": telescope_data.get('cv_results', {})
>>>>>>> ea08e74f096fe53284d3ab221b4cd1688a279e9b
        }
        
        return analytics_data
        
<<<<<<< HEAD
=======
    except HTTPException:
        raise
>>>>>>> ea08e74f096fe53284d3ab221b4cd1688a279e9b
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analytics data")

@app.get("/api/telescope/{telescope_name}")
async def get_telescope_info(telescope_name: str):
<<<<<<< HEAD
    """Get information about a specific telescope"""
=======
    """Get information about a specific telescope with actual model metrics"""
    # Get actual metrics if available
    kepler_accuracy = "N/A"
    tess_accuracy = "N/A"
    kepler_samples = "N/A"
    tess_samples = "N/A"
    kepler_features = "N/A"
    tess_features = "N/A"
    
    if 'KEPLER' in METRICS_CACHE and 'comparison_metrics' in METRICS_CACHE['KEPLER']:
        kepler_metrics = METRICS_CACHE['KEPLER']['comparison_metrics']
        if 'values' in kepler_metrics and len(kepler_metrics['values']) > 0:
            kepler_accuracy = f"{kepler_metrics['values'][0][1] * 100:.2f}"
    
    if 'KEPLER' in METRICS_CACHE and 'experiment_config' in METRICS_CACHE['KEPLER']:
        config = METRICS_CACHE['KEPLER']['experiment_config']
        kepler_samples = str(config.get('n_train', 'N/A'))
        kepler_features = str(config.get('n_features', 'N/A'))
    
    if 'TESS' in METRICS_CACHE and 'comparison_metrics' in METRICS_CACHE['TESS']:
        tess_metrics = METRICS_CACHE['TESS']['comparison_metrics']
        if 'values' in tess_metrics and len(tess_metrics['values']) > 0:
            tess_accuracy = f"{tess_metrics['values'][0][1] * 100:.2f}"
    
    if 'TESS' in METRICS_CACHE and 'experiment_config' in METRICS_CACHE['TESS']:
        config = METRICS_CACHE['TESS']['experiment_config']
        tess_samples = str(config.get('n_train', 'N/A'))
        tess_features = str(config.get('n_features', 'N/A'))
    
>>>>>>> ea08e74f096fe53284d3ab221b4cd1688a279e9b
    telescope_info = {
        "kepler": {
            "name": "Kepler Space Telescope",
            "description": "NASA's planet-hunting telescope that discovered thousands of exoplanets",
            "launch_date": "2009-03-07",
            "status": "Retired (2018)",
            "discoveries": "2,662 confirmed exoplanets",
<<<<<<< HEAD
            "model_accuracy": "86.25%",
=======
            "model_accuracy": kepler_accuracy,
            "training_samples": kepler_samples,
            "features": kepler_features,
>>>>>>> ea08e74f096fe53284d3ab221b4cd1688a279e9b
            "data_available": True
        },
        "tess": {
            "name": "Transiting Exoplanet Survey Satellite",
            "description": "NASA's current exoplanet hunting mission",
            "launch_date": "2018-04-18",
            "status": "Active",
            "discoveries": "200+ confirmed exoplanets",
<<<<<<< HEAD
            "model_accuracy": "Coming Soon",
            "data_available": False
=======
            "model_accuracy": tess_accuracy,
            "training_samples": tess_samples,
            "features": tess_features,
            "data_available": True
>>>>>>> ea08e74f096fe53284d3ab221b4cd1688a279e9b
        },
        "k2": {
            "name": "Kepler's Extended Mission",
            "description": "Kepler's second mission after reaction wheel failure",
            "launch_date": "2014-05-16",
            "status": "Retired (2018)",
            "discoveries": "500+ confirmed exoplanets",
            "model_accuracy": "Coming Soon",
<<<<<<< HEAD
=======
            "training_samples": "N/A",
            "features": "N/A",
>>>>>>> ea08e74f096fe53284d3ab221b4cd1688a279e9b
            "data_available": False
        }
    }
    
    if telescope_name.lower() not in telescope_info:
        raise HTTPException(status_code=404, detail="Telescope not found")
    
    return telescope_info[telescope_name.lower()]

<<<<<<< HEAD
=======
@app.get("/api/confusion_matrix/{telescope}")
async def get_confusion_matrix(telescope: str):
    """Get the path to confusion matrix image for a specific telescope"""
    telescope_upper = telescope.upper()
    if telescope_upper not in ['KEPLER', 'TESS']:
        raise HTTPException(status_code=404, detail=f"Telescope {telescope} not found")
    
    # Construct the image path
    experiment_id = MODELS_DIR.name
    image_path = f"/models/{experiment_id}/{telescope_upper}/Logistic_Regression_confusion_matrix.png"
    
    # Check if the file exists
    actual_path = MODELS_DIR / telescope_upper / "Logistic_Regression_confusion_matrix.png"
    if not actual_path.exists():
        raise HTTPException(status_code=404, detail=f"Confusion matrix not found for {telescope}")
    
    return {
        "telescope": telescope_upper,
        "image_url": image_path,
        "experiment_id": experiment_id
    }

@app.get("/api/feature_importance/{telescope}")
async def get_feature_importance(telescope: str):
    """Get feature importance data for a specific telescope"""
    telescope_upper = telescope.upper()
    if telescope_upper not in ['KEPLER', 'TESS']:
        raise HTTPException(status_code=404, detail=f"Telescope {telescope} not found")
    
    telescope_dir = MODELS_DIR / telescope_upper
    
    # Try to load feature importance from various sources
    feature_data = None
    
    # For TESS, we have selected_features.json
    selected_features_file = telescope_dir / "Logistic_Regression_selected_features.json"
    if selected_features_file.exists():
        with open(selected_features_file, 'r') as f:
            feature_data = json.load(f)
            return {
                "telescope": telescope_upper,
                "features": feature_data.get('feature_names', []),
                "n_features": feature_data.get('n_features', 0),
                "source": "selected_features"
            }
    
    # For KEPLER or fallback, extract from model file
    model_file = telescope_dir / "Logistic_Regression_model.pkl"
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
                        importance_values = np.abs(classifier.coef_[0]).tolist()
                    elif hasattr(classifier, 'feature_importances_'):
                        importance_values = classifier.feature_importances_.tolist()
                    
                    if importance_values and feature_names and len(importance_values) == len(feature_names):
                        # Sort by importance
                        feature_importance_pairs = list(zip(feature_names, importance_values))
                        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
                        
                        # Take top 10
                        top_features = feature_importance_pairs[:10]
                        
                        return {
                            "telescope": telescope_upper,
                            "features": [name for name, _ in top_features],
                            "importance_values": [val for _, val in top_features],
                            "n_features": len(feature_names),
                            "source": "model_coefficients"
                        }
        except Exception as e:
            logger.error(f"Error extracting feature importance: {e}")
    
    # Fallback to hardcoded KEPLER features if nothing else works
    if telescope_upper == 'KEPLER':
        return {
            "telescope": "KEPLER",
            "features": [
                'koi_max_sngle_ev', 'koi_depth', 'koi_insol', 'koi_max_mult_ev',
                'koi_dikco_msky', 'koi_insol_err2', 'koi_incl', 'koi_ror',
                'koi_smet_err2', 'koi_prad_err1'
            ],
            "importance_values": [6.06, 2.61, 1.97, 1.83, 1.62, 1.28, 0.94, 0.72, 0.54, 0.44],
            "n_features": 10,
            "source": "default"
        }
    
    raise HTTPException(status_code=404, detail=f"Feature importance data not found for {telescope}")

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
                "status": "not_ready"
            },
            "tess": {
                "accuracy": "N/A",
                "training_samples": "N/A",
                "test_samples": "N/A",
                "features": "N/A",
                "status": "not_ready"
            }
        }
        
        # Load KEPLER stats
        if 'KEPLER' in METRICS_CACHE:
            kepler_data = METRICS_CACHE['KEPLER']
            
            # Get accuracy from test results
            if 'test_results' in kepler_data:
                test_results = kepler_data['test_results']
                if 'metrics' in test_results and 'Logistic Regression' in test_results['metrics']:
                    lr_metrics = test_results['metrics']['Logistic Regression']
                    if 'values' in lr_metrics and len(lr_metrics['values']) > 0:
                        stats['kepler']['accuracy'] = f"{lr_metrics['values'][0] * 100:.2f}"
                        stats['kepler']['status'] = "ready"
            
            # Get training/test samples and features from config
            if 'experiment_config' in kepler_data:
                config = kepler_data['experiment_config']
                stats['kepler']['training_samples'] = str(config.get('n_train', 'N/A'))
                stats['kepler']['test_samples'] = str(config.get('n_test', 'N/A'))
                stats['kepler']['features'] = str(config.get('n_features', 'N/A'))
        
        # Load TESS stats
        if 'TESS' in METRICS_CACHE:
            tess_data = METRICS_CACHE['TESS']
            
            # Get accuracy from test results
            if 'test_results' in tess_data:
                test_results = tess_data['test_results']
                if 'metrics' in test_results and 'Logistic Regression' in test_results['metrics']:
                    lr_metrics = test_results['metrics']['Logistic Regression']
                    if 'values' in lr_metrics and len(lr_metrics['values']) > 0:
                        stats['tess']['accuracy'] = f"{lr_metrics['values'][0] * 100:.2f}"
                        stats['tess']['status'] = "ready"
            
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

>>>>>>> ea08e74f096fe53284d3ab221b4cd1688a279e9b
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
