from src.ml.data_prep.preprocessing import DataPreprocessor
from src.ml.data_prep.models import ModelOptimizer
from src.ml import logger
import pandas as pd
import numpy as np
from pprint import pformat


class Trainer:
    def __init__(self):
        logger.info("Trainer instance initialized")

    def trigger_training(
            self,
            filename: str,
            model_save_dir: str = None,
            cv: int = 5,
            n_iter: int = 50,
            scoring: str = 'accuracy',
            n_points: int = 10,
            n_jobs: int = -1):

        logger.info(f"Starting training process with file: {filename}")
        
        logger.info("Initializing data preprocessor")
        preprocessor = DataPreprocessor()
        
        logger.info("Running preprocessing pipeline")
        X_train, X_test, y_train, y_test = preprocessor.preprocessing_pipeline(
            filename)

        # Store feature names before converting to numpy
        if isinstance(X_train, pd.DataFrame):
            feature_names = X_train.columns.tolist()
        else:
            feature_names = None
        
        # Convert to numpy 1D array and ensure it's truly 1D
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.values.ravel()
            y_test = y_test.values.ravel()
        elif isinstance(y_train, pd.Series):
            y_train = y_train.values.ravel()
            y_test = y_test.values.ravel()
        else:
            y_train = np.asarray(y_train).ravel()
            y_test = np.asarray(y_test).ravel()

        logger.info(f"Data split completed - Train: {X_train.shape}, "
                    f"Test: {X_test.shape}")
        logger.info(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
        
        # Store data information for later access
        self.n_train = X_train.shape[0]
        self.n_test = X_test.shape[0]
        self.n_features = X_train.shape[1]
        self.n_classes = len(np.unique(y_train))
        self.feature_names = feature_names
        logger.info(f"Number of training samples: {self.n_train}")
        logger.info(f"Number of test samples: {self.n_test}")
        logger.info(f"Number of features: {self.n_features}")
        logger.info(f"Number of classes: {self.n_classes}")
        if feature_names:
            logger.info(
                f"Number of feature names stored: {len(feature_names)}")

        logger.info("Initializing model optimizer")
        model_optimizer = ModelOptimizer(model_save_dir=model_save_dir)
        
        logger.info("Training all models")
        models = model_optimizer.train_all_models(
                X_train,
                y_train,
                cv=cv,
                n_iter=n_iter,
                n_points=n_points,
                scoring=scoring,
                n_jobs=n_jobs)
        logger.info(f"Model training completed for {len(models)} models")
        
        logger.info("Evaluating models on test set")
        results = model_optimizer.evaluate_models(
            X_test, y_test, models, feature_names=feature_names)

        logger.info("Model evaluation completed")
        # logger.info(f"Training results: {results}")
        if isinstance(results, dict):
            logger.info("Training results:\n" + pformat(results, indent=4))
        else:
            logger.info("Training results:\n" + str(results))

        logger.info("Training process completed successfully")

        # Store model information for later access

        self.model_names = model_optimizer.model_names
        self.hyperparameters_spaces = model_optimizer.search_spaces
        self.model_save_dir = model_optimizer.model_save_dir
        self.training_results = results

        logger.info(f"Stored model information - Models: {self.model_names}")
        logger.info(f"Model save directory: {self.model_save_dir}")
        return results
