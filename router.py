import os
from datetime import datetime
import numpy as np
import pandas as pd
import pickle
from typing import Dict
from src.ml import logger
import json


class Router:
    """
    Router class to combine predictions from multiple trained pipelines.
    Routes predictions based on input data type (e.g., Kepler, K2, TESS).
    """

    def __init__(
            self,
            models: Dict[str, object] = None,
            weights: Dict[str, float] = None):
        """
        Initialize the Router with trained pipelines.

        Parameters:
        -----------
        models : dict, optional
            Dictionary of model pipelines, e.g.,
            {"kepler": kepler_pipeline, "k2": k2_pipeline}
            If None, models should be loaded using load_model() method
        weights : dict, optional
            Dictionary of weights for each model, e.g.,
            {"kepler": 0.6, "k2": 0.4}
            If None, equal weights are used
        """
        self.models = models or {}
        self.weights = weights or {}

        # Set default equal weights if not provided
        if self.models and not self.weights:
            self.weights = {key: 1.0 for key in self.models.keys()}

        logger.info(
            f"Router initialized with {len(self.models)} models: "
            f"{list(self.models.keys())}")
        if self.weights:
            logger.info(f"Model weights: {self.weights}")

    def load_model(self, model_name: str, model_path: str):
        """
        Load a pickled model and add it to the router.

        Parameters:
        -----------
        model_name : str
            Name/identifier for the model (e.g., "kepler", "k2", "tess")
        model_path : str
            Path to the pickled model file (.pkl)
        """
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            # Extract the trained pipeline from the saved dictionary
            if (isinstance(model_data, dict) and
                    'trained_pipeline' in model_data):
                pipeline = model_data['trained_pipeline']
                logger.info(
                    f"Loaded model '{model_name}' from {model_path}")
                logger.debug(
                    f"Model best score: "
                    f"{model_data.get('best_score', 'N/A')}")
            else:
                # If it's just the pipeline directly
                pipeline = model_data
                logger.info(
                    f"Loaded pipeline '{model_name}' from {model_path}")

            self.models[model_name] = pipeline

            # Set default weight if not already set
            if model_name not in self.weights:
                self.weights[model_name] = 1.0

        except FileNotFoundError:
            logger.error(f"Model file not found: {model_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            raise

    def set_weights(self, weights: Dict[str, float]):
        """
        Set or update weights for the models.

        Parameters:
        -----------
        weights : dict
            Dictionary of weights for each model, e.g.,
            {"kepler": 0.6, "k2": 0.4}
        """
        # Validate that all models have weights
        for model_name in self.models.keys():
            if model_name not in weights:
                logger.warning(
                    f"Weight not provided for model '{model_name}', "
                    f"using default 1.0")
                weights[model_name] = 1.0

        self.weights = weights
        logger.info(f"Updated model weights: {self.weights}")

    def predict_proba(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities using the router.

        Parameters:
        -----------
        inputs : dict
            Dictionary mapping model names to input data, e.g.,
            - Single model: {"kepler": X_kepler}
            - Multiple models: {"kepler": X_k_aligned, "k2": X_k2_aligned}
            For multiple models, data must be aligned
            (same rows/observations)

        Returns:
        --------
        np.ndarray
            Weighted average of predicted probabilities,
            shape (n_samples, n_classes)
        """
        if not inputs:
            raise ValueError("No inputs provided")

        if not self.models:
            raise ValueError("No models loaded in the router")

        probs = None
        weight_sum = 0.0
        models_used = []

        for model_name, X in inputs.items():
            model = self.models.get(model_name)

            if model is None:
                logger.warning(
                    f"Model '{model_name}' not found in router, skipping")
                continue

            # Get predictions from the model
            model_proba = model.predict_proba(X)
            weight = self.weights.get(model_name, 1.0)

            # Weighted accumulation
            if probs is None:
                probs = model_proba * weight
            else:
                probs = probs + model_proba * weight

            weight_sum += weight
            models_used.append(model_name)

            logger.debug(
                f"Used model '{model_name}' with weight {weight}")

        if probs is None:
            raise ValueError(
                f"No matching models found for provided inputs. "
                f"Available models: {list(self.models.keys())}, "
                f"Provided inputs: {list(inputs.keys())}")

        # Normalize by total weight
        probs = probs / weight_sum

        logger.debug(f"Prediction made using models: {models_used}")

        return probs

    def predict(
            self,
            inputs: Dict[str, np.ndarray],
            threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels using the router.

        Parameters:
        -----------
        inputs : dict
            Dictionary mapping model names to input data, e.g.,
            - Single model: {"kepler": X_kepler}
            - Multiple models: {"kepler": X_k_aligned, "k2": X_k2_aligned}
        threshold : float, default=0.5
            Classification threshold for binary classification

        Returns:
        --------
        np.ndarray
            Predicted class labels, shape (n_samples,)
        """
        proba = self.predict_proba(inputs)

        # For binary classification, use threshold on positive class prob
        if proba.shape[1] == 2:
            return (proba[:, 1] >= threshold).astype(int)
        else:
            # For multi-class, use argmax
            return np.argmax(proba, axis=1)
    
    def save_router(self, filepath: str):
        """
        Save the router configuration to a file.
        Note: This saves the router structure, not the full models.

        Parameters:
        -----------
        filepath : str
            Path to save the router configuration
        """
        router_config = {
            'weights': self.weights,
            'model_names': list(self.models.keys())
        }

        with open(filepath, 'wb') as f:
            pickle.dump(router_config, f)

        logger.info(f"Router configuration saved to {filepath}")
    
    @classmethod
    def load_router(cls, config_path: str, model_paths: Dict[str, str]):
        """
        Load a router from configuration and model files.

        Parameters:
        -----------
        config_path : str
            Path to the router configuration file
        model_paths : dict
            Dictionary mapping model names to their file paths,
            e.g., {"kepler": "path/to/kepler.pkl",
                   "k2": "path/to/k2.pkl"}

        Returns:
        --------
        Router
            Loaded router instance
        """
        # Load configuration
        with open(config_path, 'rb') as f:
            config = pickle.load(f)

        # Create router instance
        router = cls(weights=config.get('weights'))

        # Load models
        for model_name, model_path in model_paths.items():
            router.load_model(model_name, model_path)

        logger.info(
            f"Router loaded from {config_path} with models: "
            f"{list(router.models.keys())}")

        return router


if __name__ == "__main__":
    router = Router()
    set_weights = {}
    available_models = []

    models_dir = "./models"
    logger.info(f"Looking for models in directory: {models_dir}")
    
    if os.path.exists(models_dir):
        logger.info(f"Models directory exists: {models_dir}")
        experiment_folders = [
            f for f in os.listdir(models_dir)
            if os.path.isdir(os.path.join(models_dir, f)) and f != "cache"]
        logger.info(f"Found {len(experiment_folders)} experiment folders: "
                    f"{experiment_folders}")

        experiment_folders = ["2025.10.05_13.27.15"]
        
        if experiment_folders:
            def parse_folder_datetime(folder_name):
                try:
                    return datetime.strptime(folder_name, "%Y.%m.%d_%H.%M.%S")
                except ValueError:
                    return datetime.min
            
            # Check if there's only one experiment folder
            if len(experiment_folders) == 1:
                latest_experiment = experiment_folders[0]
                logger.info(f"Using single experiment folder: {latest_experiment}")
            else:
                latest_experiment = max(experiment_folders,
                                        key=parse_folder_datetime)
                logger.info(f"Using latest experiment folder: {latest_experiment}")
            
            for model in ["kepler", "k2"]:
                model_path = (f"./models/{latest_experiment}/"
                              f"{model.upper()}/Logistic_Regression_model.pkl")
                logger.info(f"Attempting to load {model} model from: "
                            f"{model_path}")
                
                try:
                    router.load_model(model, model_path)
                    available_models.append(model)
                    logger.info(f"Successfully loaded {model} model")
                    
                    if model == "kepler":
                        set_weights[model] = 0.6
                        logger.debug(f"Set weight for {model}: 0.6")
                    elif model == "k2":
                        set_weights[model] = 0.4
                        logger.debug(f"Set weight for {model}: 0.4")
                        
                except FileNotFoundError:
                    logger.warning(f"Model file not found: {model_path}")
                    continue
                except Exception as e:
                    logger.warning(f"Failed to load {model} model: {e}")
                    continue
        else:
            logger.error("No experiment folders found in ./models")
    else:
        logger.error("Models directory ./models not found")

    if available_models:
        logger.info(f"Setting weights for {len(available_models)} models: "
                    f"{set_weights}")
        router.set_weights(set_weights)
        logger.info("Weights successfully applied to router")
        logger.info(f"Successfully loaded models: {available_models}")
        
        # Load ALL features from preprocessed data (not just selected features)
        # The model pipeline expects the full preprocessed data and does feature selection internally
        model_data = {}
        for model in available_models:
            # Get the column names from the preprocessed training data
            from src.ml.data_prep.preprocessing import DataPreprocessor, DATATYPE
            
            if model == "kepler":
                preprocessor = DataPreprocessor(datatype=DATATYPE.KEPLER, data_dir='./data')
                # Load several samples to get column names (avoid dropping all columns)
                X, y = preprocessor.define_data('cumulative_2025.10.04_04.05.07.csv')
                X_processed = preprocessor.prepare_data(X.head(100))  # Get 100 rows to preserve columns
                
                # Create dummy data preserving the EXACT dtypes from preprocessing
                # This is critical because the pipeline expects specific dtypes (object for categorical)
                dummy_data = {}
                for col in X_processed.columns:
                    if X_processed[col].dtype == 'object':
                        # For categorical columns, use a string placeholder
                        dummy_data[col] = ['__PLACEHOLDER__']
                    else:
                        # For numeric columns, use 0.0
                        dummy_data[col] = [0.0]
                
                model_data[model] = pd.DataFrame(dummy_data)
                logger.info(f"Loaded {len(X_processed.columns)} features for {model} model")
            elif model == "k2":
                preprocessor = DataPreprocessor(datatype=DATATYPE.K2, data_dir='./data')
                # Load several samples to get column names
                X, y = preprocessor.define_data('k2pandc_2025.10.05_02.32.26.csv')
                X_processed = preprocessor.prepare_data(X.head(100))
                
                # Create dummy data preserving the EXACT dtypes from preprocessing
                dummy_data = {}
                for col in X_processed.columns:
                    if X_processed[col].dtype == 'object':
                        # For categorical columns, use a string placeholder
                        dummy_data[col] = ['__PLACEHOLDER__']
                    else:
                        # For numeric columns, use 0.0
                        dummy_data[col] = [0.0]
                
                model_data[model] = pd.DataFrame(dummy_data)
                logger.info(f"Loaded {len(X_processed.columns)} features for {model} model")
        
        # Make predictions with loaded models using dummy data
        if "kepler" in available_models and "k2" in available_models:
            logger.info("Making ensemble predictions with both models")
            y_pred_ensemble = router.predict({
                "kepler": model_data["kepler"],
                "k2": model_data["kepler"]
            })
            y_proba_ensemble = router.predict_proba({
                "kepler": model_data["kepler"],
                "k2": model_data["kepler"]
            })
            logger.info(f"Ensemble predictions: {y_pred_ensemble}")
            logger.info(f"Ensemble probabilities: {y_proba_ensemble}")
        elif "kepler" in available_models:
            logger.info("Making predictions with Kepler model only")
            y_pred_kepler = router.predict({"kepler": model_data["kepler"]})
            y_proba_kepler = router.predict_proba({"kepler": model_data["kepler"]})
            logger.info(f"Kepler predictions: {y_pred_kepler}")
            logger.info(f"Kepler probabilities: {y_proba_kepler}")
        elif "k2" in available_models:
            logger.info("Making predictions with K2 model only")
            y_pred_k2 = router.predict({"k2": model_data["k2"]})
            y_proba_k2 = router.predict_proba({"k2": model_data["k2"]})
            logger.info(f"K2 predictions: {y_pred_k2}")
            logger.info(f"K2 probabilities: {y_proba_k2}")
    else:
        logger.error("No models could be loaded")
