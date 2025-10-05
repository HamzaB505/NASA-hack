import pandas as pd
from pprint import pformat  # Add this to imports
import pickle
import os
import json
import numpy as np
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from joblib import Memory
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from tqdm import tqdm
from src.ml import logger
from src.ml.data_prep.metrics import (compute_and_show_confusion_matrix,
                                      get_classification_metrics,
                                      compare_models_metrics)
from src.ml.data_prep.utils import (extract_feature_importance,
                                    extract_selected_features,
                                    map_transformed_to_original_features)


class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Custom transformer to cap outliers at specified percentiles.
    Preserves NaN values for subsequent imputation.
    """
    
    def __init__(self, percentile=0.01):
        """
        Initialize the OutlierCapper.
        
        Parameters:
        -----------
        percentile : float, default=0.01
            Percentile threshold for capping (0.01 means cap at 1st and 99th)
        """
        self.percentile = percentile
        self.bounds_ = {}
    
    def fit(self, X, y=None):
        """
        Learn the outlier bounds from the training data.
        
        Parameters:
        -----------
        X : array-like or pd.DataFrame
            Input features
        y : array-like, optional
            Target variable (not used)
            
        Returns:
        --------
        self
        """
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        outliers_count = 0
        n_features_with_outliers = 0
        
        for col_idx, col in enumerate(X_df.columns):
            col_data = X_df.iloc[:, col_idx] if isinstance(X_df.columns[0], int) else X_df[col]
            
            if col_data.notna().sum() > 0:  # Only if has non-NaN values
                lower_bound = col_data.quantile(self.percentile)
                upper_bound = col_data.quantile(1 - self.percentile)
                self.bounds_[col_idx] = {
                    'lower': lower_bound,
                    'upper': upper_bound
                }
                
                # Count outliers (both tails)
                outliers_lower = (col_data < lower_bound).sum()
                outliers_upper = (col_data > upper_bound).sum()
                feature_outliers = outliers_lower + outliers_upper
                
                if feature_outliers > 0:
                    n_features_with_outliers += 1
                    outliers_count += feature_outliers
                    
                    # Get feature name
                    feature_name = col if not isinstance(col, int) else f"Feature_{col}"
                    
                    # Log per-feature details
                    cap_details = []
                    if outliers_lower > 0:
                        cap_details.append(
                            f"{outliers_lower} values < {lower_bound:.4f} "
                            f"capped to {lower_bound:.4f}"
                        )
                    if outliers_upper > 0:
                        cap_details.append(
                            f"{outliers_upper} values > {upper_bound:.4f} "
                            f"capped to {upper_bound:.4f}"
                        )
                    
                    logger.info(f"  Feature '{feature_name}': {', '.join(cap_details)}")
        
        if outliers_count > 0:
            logger.info(f"OutlierCapper: Total {outliers_count} outliers capped across "
                       f"{n_features_with_outliers} features")
        
        return self
    
    def transform(self, X):
        """
        Apply outlier capping to the input data.
        
        Parameters:
        -----------
        X : array-like or pd.DataFrame
            Input features
            
        Returns:
        --------
        np.ndarray
            Transformed features with capped outliers
        """
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        for col_idx, bounds in self.bounds_.items():
            if col_idx < len(X_df.columns):
                col_data = X_df.iloc[:, col_idx] if isinstance(X_df.columns[0], int) else X_df[X_df.columns[col_idx]]
                X_df.iloc[:, col_idx] = col_data.clip(
                    lower=bounds['lower'],
                    upper=bounds['upper']
                )
        
        return X_df.values if isinstance(X, np.ndarray) else X_df.values


class NaNPreservingOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    Custom OneHotEncoder that preserves NaN values in the encoded output.
    This allows KNNImputer to impute categorical features alongside numerical ones.
    """
    
    def __init__(self, handle_unknown='ignore', drop='first', sparse_output=False, dtype='float64'):
        """
        Initialize the NaNPreservingOneHotEncoder.
        
        Parameters:
        -----------
        handle_unknown : str, default='ignore'
            How to handle unknown categories during transform
        drop : str, default='first'
            Whether to drop the first category to avoid multicollinearity
        sparse_output : bool, default=False
            Whether to return sparse matrix
        dtype : str, default='float64'
            Data type for output
        """
        self.handle_unknown = handle_unknown
        self.drop = drop
        self.sparse_output = sparse_output
        self.dtype = dtype
        self.encoder = None
        self.n_features_in_ = None
        self.feature_names_out_ = None
    
    def fit(self, X, y=None):
        """
        Fit the OneHotEncoder on non-NaN values.
        
        Parameters:
        -----------
        X : array-like or pd.DataFrame
            Input categorical features
        y : array-like, optional
            Target variable (not used)
            
        Returns:
        --------
        self
        """
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        self.n_features_in_ = X_df.shape[1]
        
        # Create temporary data with NaN replaced by a sentinel for fitting
        X_temp = X_df.copy()
        sentinel = '__NAN_SENTINEL__'
        
        for col in X_temp.columns:
            X_temp[col] = X_temp[col].fillna(sentinel)
        
        # Fit the encoder on non-NaN values
        self.encoder = OneHotEncoder(
            handle_unknown=self.handle_unknown,
            drop=self.drop,
            sparse_output=self.sparse_output,
            dtype=self.dtype
        )
        self.encoder.fit(X_temp)
        
        # Store feature names for reference
        try:
            self.feature_names_out_ = self.encoder.get_feature_names_out()
        except AttributeError:
            self.feature_names_out_ = None
        
        logger.debug(f"NaNPreservingOneHotEncoder fitted with {self.n_features_in_} input features, "
                    f"producing {len(self.feature_names_out_) if self.feature_names_out_ is not None else 'unknown'} output features")
        
        return self
    
    def transform(self, X):
        """
        Transform categorical features to one-hot encoding, preserving NaN as NaN.
        
        Parameters:
        -----------
        X : array-like or pd.DataFrame
            Input categorical features
            
        Returns:
        --------
        np.ndarray
            One-hot encoded features with NaN preserved
        """
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # Track which rows have NaN in each column
        nan_masks = []
        for col_idx, col in enumerate(X_df.columns):
            nan_mask = X_df.iloc[:, col_idx].isna()
            nan_masks.append(nan_mask)
        
        # Replace NaN with sentinel for encoding
        X_temp = X_df.copy()
        sentinel = '__NAN_SENTINEL__'
        for col in X_temp.columns:
            X_temp[col] = X_temp[col].fillna(sentinel)
        
        # Apply one-hot encoding (suppress unknown category warnings)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',
                                    message='Found unknown categories',
                                    category=UserWarning)
            X_encoded = self.encoder.transform(X_temp)
        
        # Convert to dense array if needed
        if hasattr(X_encoded, 'toarray'):
            X_encoded = X_encoded.toarray()
        
        # Get the feature ranges for each original column
        feature_ranges = []
        start_idx = 0
        for cat_idx, categories in enumerate(self.encoder.categories_):
            n_categories = len(categories)
            if self.drop == 'first':
                n_categories -= 1
            feature_ranges.append((start_idx, start_idx + n_categories))
            start_idx += n_categories
        
        # Set encoded features to NaN where original was NaN
        total_nan_rows = 0
        for col_idx, nan_mask in enumerate(nan_masks):
            if nan_mask.any():
                start_idx, end_idx = feature_ranges[col_idx]
                X_encoded[nan_mask, start_idx:end_idx] = np.nan
                total_nan_rows += nan_mask.sum()
        
        if total_nan_rows > 0:
            logger.debug(f"Preserved {total_nan_rows} NaN values across "
                        f"{len(nan_masks)} categorical features for KNN imputation")
        
        return X_encoded


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
        Each pipeline uses ColumnTransformer to handle numerical and categorical
        features separately, followed by KNN imputation, feature selection, 
        and a classifier.
        
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
                random_state=self.random_state)
            # ),
            # 'Decision Tree': DecisionTreeClassifier(
            #     random_state=self.random_state
            # ),
            # 'Random Forest': RandomForestClassifier(
            #     random_state=self.random_state
            # ),
            # 'Gradient Boosting': GradientBoostingClassifier(
            #     random_state=self.random_state
            # ),
            # 'XGBoost': XGBClassifier(random_state=self.random_state),
            # 'SVM': SVC(random_state=self.random_state, probability=True)
        }
        
        logger.info(f"Creating pipelines for {len(models)} models: {list(models.keys())}")
        
        # Create pipelines for each model
        pipelines = {}
        
        for model_name, model in models.items():
            # Define numerical processor: cap outliers only (no imputation/scaling yet)
            numeric_processor = Pipeline(steps=[
                ("outlier_capper", OutlierCapper(percentile=0.01))
            ])
            
            # Define categorical processor: one-hot encode while preserving NaN
            categorical_processor = Pipeline(steps=[
                ("onehot", NaNPreservingOneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    sparse_output=False,
                    dtype='float64'
                ))
            ])
            
            # Create ColumnTransformer to apply processors to respective columns
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_processor, 
                     make_column_selector(dtype_include=np.number)),
                    ("cat", categorical_processor, 
                     make_column_selector(dtype_include=object))
                ]
            )
            
            # Wrap the model with calibration
            calibrated_model = CalibratedClassifierCV(
                estimator=model,
                method='sigmoid',  # Use sigmoid (Platt scaling) for binary classification
                cv=3,  # Use 3-fold CV for calibration
                n_jobs=-1
            )
            
            # Create the full pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('imputer', KNNImputer(n_neighbors=5)),
                ('scaler', StandardScaler()),
                ('feature_selector', SelectFromModel(
                    RandomForestClassifier(
                        n_estimators=100,
                        random_state=self.random_state)
                )),
                ('classifier', calibrated_model)
                ],
                memory=self.memory)
            pipelines[model_name] = pipeline
            logger.debug(f"Created pipeline for {model_name} with probability calibration")
        
        self.pipelines = pipelines
        logger.info(f"Successfully created {len(pipelines)} model pipelines")
        return pipelines

    def get_bayesian_search_spaces(self):
        """
        Define hyperparameter search spaces for Bayesian optimization.
        Note: classifier is wrapped with CalibratedClassifierCV, so parameters
        need to use 'classifier__estimator__' prefix to access the base estimator.
        
        Returns:
        --------
        dict
            Dictionary containing model name as key and search space as value
        """
        logger.info("Defining Bayesian search spaces for hyperparameter optimization")
        
        search_spaces = {
            'Logistic Regression': {
                'feature_selector__max_features': Integer(5, 30),
                'classifier__estimator__C': Real(0.01, 100, prior='log-uniform'),
                'classifier__estimator__penalty': Categorical(['l2'])
            },
            
            'Random Forest': {
                'feature_selector__max_features': Integer(5, 30),
                'classifier__estimator__n_estimators': Integer(50, 300),
                'classifier__estimator__min_samples_split': Integer(2, 20)
            },
            
            'Gradient Boosting': {
                'feature_selector__max_features': Integer(5, 30),
                'classifier__estimator__n_estimators': Integer(50, 300),
                'classifier__estimator__learning_rate': Real(
                    0.01, 0.3, prior='log-uniform'
                ),
                'classifier__estimator__max_depth': Integer(3, 10)
            },
            
            'XGBoost': {
                'feature_selector__max_features': Integer(5, 30),
                'classifier__estimator__n_estimators': Integer(50, 300),
                'classifier__estimator__learning_rate': Real(
                    0.01, 0.3, prior='log-uniform'
                ),
                'classifier__estimator__max_depth': Integer(3, 10)
            },
            
            'SVM': {
                'feature_selector__max_features': Integer(5, 30),
                'classifier__estimator__C': Real(0.1, 100, prior='log-uniform')
            },
            
            'Decision Tree': {
                'feature_selector__max_features': Integer(5, 30),
                'classifier__estimator__min_samples_split': Integer(2, 20)
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
                n_points=10,
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
        n_points : int, default=10
            Number of points to sample from the search space
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
            n_points=n_points,
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
            "description": "Best cross-validation results",
            "best_score": float(pipeline_result["best_score"]),
            "best_params": pipeline_result["best_params"],
            "cv_results": {
                k: v.tolist() if hasattr(v, 'tolist') else v 
                for k, v in pipeline_result["cv_results"].items()}
        }
        
        results_path = os.path.join(self.model_save_dir, 
                                    f'{clean_name}_cv_results.json')
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
                "description": "Best cross-validation results",
                "best_score": pipeline_result["best_score"],
                "best_params": pipeline_result["best_params"],
                "cv_results": pipeline_result["cv_results"]
            }
        
        results_path = os.path.join(self.model_save_dir,
                                    'validation_results.json')
        with open(results_path, 'wb') as f:
            pickle.dump(results_summary, f)
        
        logger.info(f"Models and results saved to {self.model_save_dir}")

    def train_all_models(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            cv: int = 5,
            n_iter: int = 50,
            n_points: int = 10,
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
                n_points=n_points,
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
        self.model_names = model_names
        logger.info(f"Completed training of all {len(optimized_models)} models")

        return optimized_models

    def evaluate_models(
            self,
            X_test: pd.DataFrame,
            y_test: pd.Series,
            models: dict,
            feature_names: list = None):
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
        feature_names : list, optional
            List of feature names for feature importance extraction
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
                "y_test": y_test,
                "y_pred": y_pred,
                "y_pred_proba": y_pred_proba
            }

            logger.info(f"Confusion matrix for {model_name}:")
            cf_df = compute_and_show_confusion_matrix(y_test, y_pred, model_name, save_dir=self.model_save_dir)
            results[model_name]["confusion_matrix"] = cf_df.to_dict()

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
            
            # Extract and save selected features for prediction
            if feature_names is not None:
                logger.info(
                    f"Extracting selected features for {model_name}")
                selected_features = extract_selected_features(
                    models[model_name], feature_names, model_name)
                
                if selected_features is not None:
                    # Create mapping from transformed to original feature names
                    feature_mapping = map_transformed_to_original_features(selected_features)
                    
                    # Save selected features to JSON
                    features_json = {
                        "n_features": len(selected_features),
                        "feature_names": selected_features,
                        "feature_mapping": feature_mapping,
                        "note": "These are the exact transformed feature names required for prediction. "
                               "Use 'feature_mapping' to understand which original features they came from."
                    }
                    features_json_path = os.path.join(
                        self.model_save_dir,
                        f'{clean_name}_selected_features.json')
                    with open(features_json_path, 'w') as f:
                        json.dump(features_json, f, indent=2)
                    logger.info(
                        f"Saved {len(selected_features)} selected features to {features_json_path}")
                    
                    # Also save a simplified version with just original feature names
                    # (for backward compatibility and easier reading)
                    original_features = list(set([
                        mapping['original_feature'] 
                        for mapping in feature_mapping.values()
                    ]))
                    simple_features_json = {
                        "n_original_features": len(original_features),
                        "original_feature_names": sorted(original_features),
                        "note": "These are the original feature names (before transformations like one-hot encoding). "
                               "For the exact transformed feature names the model expects, see the main selected_features file."
                    }
                    simple_features_json_path = os.path.join(
                        self.model_save_dir,
                        f'{clean_name}_original_features.json')
                    with open(simple_features_json_path, 'w') as f:
                        json.dump(simple_features_json, f, indent=2)
                    logger.info(
                        f"Saved {len(original_features)} original feature names to {simple_features_json_path}")
                else:
                    logger.warning(
                        f"Could not extract selected features for {model_name}")
            
            # Extract and save feature importance
            if feature_names is not None:
                logger.info(
                    f"Extracting feature importance for {model_name}")
                importance_df = extract_feature_importance(
                    models[model_name], feature_names, model_name)

                if importance_df is not None:
                    # Log top 10 features
                    logger.info(
                        f"Top 10 most important features for {model_name}:")
                    top_10 = importance_df.head(10)
                    for idx, row in top_10.iterrows():
                        logger.info(
                            f"  {idx+1}. {row['feature_name']}: "
                            f"{row['importance']:.6f}")

                    # Save to CSV
                    importance_csv_path = os.path.join(
                        self.model_save_dir,
                        f'{clean_name}_feature_importance.csv')
                    importance_df.to_csv(importance_csv_path, index=False)
                    logger.info(
                        f"Saved feature importance to {importance_csv_path}")

                    # Save to JSON
                    importance_json = {
                        "feature_names":
                            importance_df['feature_name'].tolist(),
                        "importance_values":
                            importance_df['importance'].tolist()
                    }
                    importance_json_path = os.path.join(
                        self.model_save_dir,
                        f'{clean_name}_feature_importance.json')
                    with open(importance_json_path, 'w') as f:
                        json.dump(importance_json, f, indent=2)
                    logger.info(
                        f"Saved feature importance JSON to "
                        f"{importance_json_path}")

                    # Store in results
                    results[model_name]["feature_importance"] = (
                        importance_df.to_dict('records'))
                else:
                    logger.info(
                        f"Model {model_name} does not support "
                        f"feature importance extraction")
            
            logger.info(f"Completed evaluation for {model_name}")
        
        # Remove predictions from results before returning
        # (keep metrics, confusion matrix, and feature importance)
        for model_name in results.keys():
            result_dict = {
                "metrics": results[model_name]["metrics"],
                "confusion_matrix": results[model_name]["confusion_matrix"]
            }
            if "feature_importance" in results[model_name]:
                result_dict["feature_importance"] = (
                    results[model_name]["feature_importance"])
            results[model_name] = result_dict

        logger.info(f"Completed evaluation of all {len(results)} models")
        return results
