import pandas as pd
from pprint import pformat  # Add this to imports
import pickle
import os
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold
from joblib import Memory
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from tqdm import tqdm
from src.ml import logger
from src.ml.data_prep.metrics import (compute_and_show_confusion_matrix, 
                                      get_classification_metrics,
                                      compare_models_metrics)


class ModelOptimizer:
    """
    A class for creating and optimizing machine learning model pipelines 
    using Bayesian optimization.
    """
    
    def __init__(self, random_state=42, model_save_dir=None):
        """
        Initialize the ModelOptimizer.
        
        Parameters:
        -----------
        random_state : int, default=42
            Random state for reproducibility across all models and operations
        model_save_dir : str, default=None
            Directory to save trained models and results
        """
        if model_save_dir is None:
            model_save_dir = "/Users/hamzaboulaala/Documents/github/NASA-hack/models"

        self.random_state = random_state
        self.model_save_dir = model_save_dir
        
        # Use a shared cache directory to avoid duplication across timestamped runs
        cache_dir = "/Users/hamzaboulaala/Documents/github/NASA-hack/models/cache"
        self.memory = Memory(
                    location=cache_dir,
                    compress=True,
                    verbose=0)

        self.pipelines = None
        self.search_spaces = None
        self.optimized_models = {}
        logger.info(f"ModelOptimizer initialized with random_state={random_state}, model_save_dir={model_save_dir}")
    
    def create_model_pipelines(self):
        """
        Create a collection of model pipelines with standard preprocessing.
        Each pipeline includes KNN imputation, standard scaling, feature 
        selection, and a classifier.
        
        Returns:
        --------
        dict
            Dictionary containing model name as key and pipeline as value
        """
        logger.info("Creating model pipelines")
        
        # Define the models to include
        models = {
            'Logistic Regression': LogisticRegression(
                solver='liblinear',
                random_state=self.random_state
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=self.random_state
            ),
            'Random Forest': RandomForestClassifier(
                random_state=self.random_state
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=self.random_state
            ),
            'XGBoost': XGBClassifier(random_state=self.random_state),
            'SVM': SVC(random_state=self.random_state, probability=True),
            'Perceptron': Perceptron(random_state=self.random_state),
        }
        
        logger.info(f"Creating pipelines for {len(models)} models: {list(models.keys())}")
        
        # Create pipelines for each model
        pipelines = {}
        
        for model_name, model in models.items():
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('imputer', KNNImputer(n_neighbors=5)),
                ('feature_selector', SelectFromModel(
                    RandomForestClassifier(
                        n_estimators=100,
                        random_state=self.random_state)
                )),
                ('classifier', model)
                ],
                memory=self.memory)
            pipelines[model_name] = pipeline
            logger.debug(f"Created pipeline for {model_name}")
        
        self.pipelines = pipelines
        logger.info(f"Successfully created {len(pipelines)} model pipelines")
        return pipelines

    def get_bayesian_search_spaces(self):
        """
        Define hyperparameter search spaces for Bayesian optimization.
        
        Returns:
        --------
        dict
            Dictionary containing model name as key and search space as value
        """
        logger.info("Defining Bayesian search spaces for hyperparameter optimization")
        
        search_spaces = {
            'Logistic Regression': {
                'feature_selector__max_features': Integer(5, 30),
                'classifier__C': Real(0.01, 100, prior='log-uniform'),
                'classifier__penalty': Categorical(['l1', 'l2'])
            },
            
            'Random Forest': {
                'feature_selector__max_features': Integer(5, 30),
                'classifier__n_estimators': Integer(50, 300),
                'classifier__min_samples_split': Integer(2, 20)
            },
            
            'Gradient Boosting': {
                'feature_selector__max_features': Integer(5, 30),
                'classifier__n_estimators': Integer(50, 300),
                'classifier__learning_rate': Real(
                    0.01, 0.3, prior='log-uniform'
                ),
                'classifier__max_depth': Integer(3, 10)
            },
            
            'XGBoost': {
                'feature_selector__max_features': Integer(5, 30),
                'classifier__n_estimators': Integer(50, 300),
                'classifier__learning_rate': Real(
                    0.01, 0.3, prior='log-uniform'
                ),
                'classifier__max_depth': Integer(3, 10)
            },
            
            'SVM': {
                'feature_selector__max_features': Integer(5, 30),
                'classifier__C': Real(0.1, 100, prior='log-uniform')
            },
            
            'Decision Tree': {
                'feature_selector__max_features': Integer(5, 30),
                'classifier__min_samples_split': Integer(2, 20)
            },
            'Perceptron': {
                'feature_selector__max_features': Integer(5, 30),
                'classifier__alpha': Real(0.0001, 0.1, prior='log-uniform')
            }
        }
        
        self.search_spaces = search_spaces
        logger.info(f"Defined search spaces for {len(search_spaces)} models")
        return search_spaces

    def optimize_model_with_bayesian_search(
                self,
                pipeline,
                search_space,
                X_train,
                y_train,
                cv=5,
                n_iter=50,
                scoring='accuracy',
                n_jobs=-1):

        """
        Optimize a model pipeline using Bayesian optimization with Stratified K-Fold.
        
        Parameters:
        -----------
        pipeline : sklearn.pipeline.Pipeline
            The pipeline to optimize
        search_space : dict
            Dictionary defining the hyperparameter search space
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        cv : int, default=5
            Number of cross-validation folds for StratifiedKFold
        n_iter : int, default=50
            Number of optimization iterations
        scoring : str, default='accuracy'
            Scoring metric for optimization
        n_jobs : int, default=-1
            Number of parallel jobs
        
        Returns:
        --------
        BayesSearchCV
            Fitted Bayesian search object with optimized hyperparameters
        """
        # Create StratifiedKFold cross-validator
        stratified_cv = StratifiedKFold(
            n_splits=cv, shuffle=True, random_state=42
        )
        
        logger.debug(
            f"Starting Bayesian optimization with {n_iter} iterations, "
            f"StratifiedKFold(n_splits={cv}, shuffle=True, "
            f"random_state=42), scoring={scoring}"
        )
        
        bayes_search = BayesSearchCV(
            estimator=pipeline,
            search_spaces=search_space,
            n_iter=n_iter,
            cv=stratified_cv,
            n_points=10,
            scoring=scoring,
            random_state=self.random_state,
            n_jobs=n_jobs,
            refit=True,
            verbose=1
        )
        # Ensure y_train is a 1D numpy array
        if isinstance(y_train, (pd.Series, pd.DataFrame)):
            y_train = y_train.values.ravel()
        else:
            y_train = np.asarray(y_train).ravel()

        logger.debug(f"y_train shape after conversion: {y_train.shape}")

        bayes_search.fit(X_train, y_train)
        logger.debug(f"Bayesian optimization completed with best score: {bayes_search.best_score_:.4f}")

        pipeline_results = {
            "trained_pipeline": bayes_search.best_estimator_,
            "best_score": bayes_search.best_score_,
            "best_params": bayes_search.best_params_,
            "cv_results": bayes_search.cv_results_
        }
        return pipeline_results

    def save_model_and_result(self, model_name, pipeline_result):
        """
        Save a single trained model and its validation results to disk.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        pipeline_result : dict
            Dictionary containing trained_pipeline, best_score, best_params, cv_results
        """
        logger.info(f"Saving model {model_name} to {self.model_save_dir}")
        
        # Create directory if it doesn't exist
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        # Clean model name for filename
        clean_name = model_name.replace(' ', '_').replace('/', '_')
        model_path = os.path.join(self.model_save_dir, 
                                  f'{clean_name}_model.pkl')
        
        # Save the entire pipeline result
        with open(model_path, 'wb') as f:
            pickle.dump(pipeline_result, f)
        logger.debug(f"Saved model {model_name} to {model_path}")
        
        # Save individual result summary as JSON
        result_summary = {
            "best_score": float(pipeline_result["best_score"]),
            "best_params": pipeline_result["best_params"],
            "cv_results": {k: v.tolist() if hasattr(v, 'tolist') else v 
                          for k, v in pipeline_result["cv_results"].items()}
        }
        
        results_path = os.path.join(self.model_save_dir, 
                                    f'{clean_name}_results.json')
        import json
        with open(results_path, 'w') as f:
            json.dump(result_summary, f, indent=2)
        
        # logger.info(f"Model {model_name} and results saved")
        logger.info(f"Model {model_name} saved to {model_path}")

    def save_models_and_results(self, pipeline_results):
        """
        Save trained models and their validation results to disk.
        Parameters:
        -----------
        pipeline_results : dict
            Dictionary containing model name as key and optimized 
            BayesSearchCV object as value
        """
        logger.info(f"Saving {len(pipeline_results)} models and results to {self.model_save_dir}")
        
        # Create directory if it doesn't exist
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        # Save each optimized model
        for model_name, pipeline_result in pipeline_results.items():
            # Clean model name for filename
            clean_name = model_name.replace(' ', '_').replace('/', '_')
            model_path = os.path.join(self.model_save_dir, 
                                      f'{clean_name}_model.pkl')
            
            # Save the entire BayesSearchCV object
            with open(model_path, 'wb') as f:
                pickle.dump(pipeline_result, f)
            logger.debug(f"Saved model {model_name} to {model_path}")
        
        # Save validation results summary
        results_summary = {}
        for model_name, pipeline_result in pipeline_results.items():
            results_summary[model_name] = {
                "best_score": pipeline_result["best_score"],
                "best_params": pipeline_result["best_params"],
                "cv_results": pipeline_result["cv_results"]
            }
        
        results_path = os.path.join(self.model_save_dir, 
                                    'validation_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(results_summary, f)
        
        logger.info(f"Models and results saved to {self.model_save_dir}")

    def train_all_models(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            cv: int = 5,
            n_iter: int = 50,
            scoring: str = 'accuracy',
            n_jobs: int = -1):
        """
        Optimize all model pipelines using Bayesian optimization.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training labels
        cv : int, default=5
            Number of cross-validation folds
        n_iter : int, default=50
            Number of optimization iterations per model
        scoring : str, default='accuracy'
            Scoring metric for optimization
        n_jobs : int, default=-1
            Number of parallel jobs

        Returns:
        --------
        dict
            Dictionary containing model name as key and optimized 
            BayesSearchCV object as value
        """
        logger.info(f"Starting training of all models with parameters: cv={cv}, n_iter={n_iter}, scoring={scoring}")

        if self.pipelines is None:
            self.create_model_pipelines()
        if self.search_spaces is None:
            self.get_bayesian_search_spaces()

        optimized_models = {}
        model_names = list(self.pipelines.keys())
        logger.info(f"Training {len(model_names)} models: {model_names}")

        # Use tqdm for progress tracking
        for model_name in tqdm(model_names, desc="Training models", unit="model"):
            logger.info(f"Starting training for {model_name}")
            logger.info(f"\nOptimizing {model_name}...")

            pipeline_results = self.optimize_model_with_bayesian_search(
                pipeline=self.pipelines[model_name],
                search_space=self.search_spaces[model_name],
                X_train=X_train,
                y_train=y_train,
                cv=cv,
                n_iter=n_iter,
                scoring=scoring,
                n_jobs=n_jobs
            )
            
            # Save immediately after training completes
            self.save_model_and_result(model_name, pipeline_results)
            
            optimized_models[model_name] = pipeline_results["trained_pipeline"]

            logger.info(f"Completed optimization for {model_name} - Best score: {pipeline_results['best_score']:.4f}")
            logger.info(f"Best score for {model_name}: "
                  f"{pipeline_results['best_score']:.4f}")
            logger.info(f"Best parameters for {model_name}: "
                  f"{pipeline_results['best_params']}")

        self.optimized_models = optimized_models
        logger.info(f"Completed training of all {len(optimized_models)} models")

        return optimized_models

    def evaluate_models(
            self,
            X_test: pd.DataFrame,
            y_test: pd.Series,
            models: dict):
        """
        Evaluate a list of models on the test set.
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test labels
        models : dict
            Dictionary containing model name as key and model as value
        """
        logger.info(f"Starting evaluation of {len(models)} models on test set")

        results = {}
        model_names = list(models.keys())

        # Use tqdm for progress tracking during evaluation
        for model_name in tqdm(model_names, desc="Evaluating models", unit="model"):
            logger.info(f"Evaluating {model_name}")
            logger.info(f"\nEvaluating {model_name}...")
            
            # Use the trained models passed as parameter, not self.pipelines
            y_pred = models[model_name].predict(X_test)
            y_pred_proba = models[model_name].predict_proba(X_test)

            # Store predictions temporarily for metrics computation (not saved to results)
            results[model_name] = {
                "y_test": list(y_test),
                "y_pred": list(y_pred),
                "y_pred_proba": list(y_pred_proba)
            }

            logger.info(f"Confusion matrix for {model_name}:")
            compute_and_show_confusion_matrix(y_test, y_pred, model_name, save_dir=self.model_save_dir)

            logger.info(f"Metrics for {model_name}:")
            metrics = get_classification_metrics(y_test, y_pred, y_pred_proba, model_name)
            results[model_name]["metrics"] = metrics
            logger.info("\n" + pformat(metrics, indent=4))

            logger.info(f"Comparison of metrics for {model_name}:")
            comparison_df = compare_models_metrics(results)
            # logger.info("\n" + pformat(comparison_df, indent=4))

            # Clean model name for filename
            clean_name = model_name.replace(' ', '_').replace('/', '_')
            comparison_df.to_csv(
                os.path.join(self.model_save_dir, f'{clean_name}_comparison_metrics.csv'), index=False)

            # Save metrics as JSON with columns and values as separate lists
            metrics_json = {
                "columns": comparison_df.columns.tolist(),
                "values": comparison_df.values.tolist()
            }
            
            metrics_json_path = os.path.join(self.model_save_dir, f'{clean_name}_comparison_metrics.json')
            with open(metrics_json_path, 'w') as f:
                json.dump(metrics_json, f, indent=2)
            
            logger.info(f"Saved metrics to {metrics_json_path}")
            logger.info(f"Completed evaluation for {model_name}")
        
        # Remove predictions from results before returning (only keep metrics)
        for model_name in results.keys():
            results[model_name] = {
                "metrics": results[model_name]["metrics"]
            }

        logger.info(f"Completed evaluation of all {len(results)} models")
        return results
