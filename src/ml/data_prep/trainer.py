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
            n_jobs: int = -1):

        logger.info(f"Starting training process with file: {filename}")
        
        logger.info("Initializing data preprocessor")
        preprocessor = DataPreprocessor()
        
        logger.info("Running preprocessing pipeline")
        X_train, X_test, y_train, y_test = preprocessor.preprocessing_pipeline(
            filename)

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
        
        logger.info("Initializing model optimizer")
        model_optimizer = ModelOptimizer(model_save_dir=model_save_dir)
        
        logger.info("Training all models")
        models = model_optimizer.train_all_models(
                X_train,
                y_train,
                cv=cv,
                n_iter=n_iter,
                scoring=scoring,
                n_jobs=n_jobs)
        logger.info(f"Model training completed for {len(models)} models")
        
        logger.info("Evaluating models on test set")
        results = model_optimizer.evaluate_models(X_test, y_test, models)

        logger.info("Model evaluation completed")
        # logger.info(f"Training results: {results}")
        if isinstance(results, dict):
            logger.info("Training results:\n" + pformat(results, indent=4))
        else:
            logger.info("Training results:\n" + str(results))

        logger.info("Training process completed successfully")
        return results
