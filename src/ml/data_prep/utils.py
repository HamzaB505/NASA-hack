import numpy as np
import pandas as pd
from src.ml import logger


def get_transformed_feature_names(pipeline, original_feature_names):
    """
    Get the feature names after preprocessing transformations (one-hot encoding, etc.)
    but before feature selection.
    
    Parameters:
    -----------
    pipeline : sklearn.pipeline.Pipeline
        Trained pipeline with preprocessor step
    original_feature_names : list
        Original feature names before transformations
    
    Returns:
    --------
    list
        Transformed feature names after preprocessing
    """
    try:
        # Get the preprocessor (ColumnTransformer) from the pipeline
        preprocessor = pipeline.named_steps.get('preprocessor', None)
        
        if preprocessor is None:
            logger.warning("No preprocessor found in pipeline, using original feature names")
            return original_feature_names
        
        # Get transformed feature names from ColumnTransformer
        if hasattr(preprocessor, 'get_feature_names_out'):
            # This method is available in newer sklearn versions
            transformed_names = preprocessor.get_feature_names_out()
            return list(transformed_names)
        else:
            logger.warning("Could not get transformed feature names, using original names")
            return original_feature_names
    
    except Exception as e:
        logger.error(f"Error getting transformed feature names: {str(e)}")
        return original_feature_names


def map_transformed_to_original_features(transformed_feature_names):
    """
    Map transformed feature names back to original feature names.
    This helps understand which original features the transformed features came from.
    
    Parameters:
    -----------
    transformed_feature_names : list
        List of transformed feature names (e.g., "num__koi_depth", "cat__comment_str_CENTROID")
    
    Returns:
    --------
    dict
        Dictionary mapping transformed names to original feature names
    """
    feature_mapping = {}
    
    for transformed_name in transformed_feature_names:
        # ColumnTransformer prefixes features with transformer name
        # Format: "transformer_name__feature_name" or "transformer_name__column_category"
        
        if '__' in transformed_name:
            # Split on the first occurrence of '__'
            parts = transformed_name.split('__', 1)
            if len(parts) == 2:
                transformer_type, feature_part = parts
                
                # For categorical features that were one-hot encoded,
                # the format is typically: cat__original_column_name_category_value
                # For numerical features: num__original_column_name
                
                # Extract the base feature name (before any category suffix)
                # This is a heuristic - for one-hot encoded features, the original
                # column name is before the last underscore(s) representing the category
                original_name = feature_part
                
                feature_mapping[transformed_name] = {
                    'original_feature': original_name,
                    'transformer_type': transformer_type,
                    'is_encoded': transformer_type == 'cat'
                }
            else:
                feature_mapping[transformed_name] = {
                    'original_feature': transformed_name,
                    'transformer_type': 'unknown',
                    'is_encoded': False
                }
        else:
            # No transformation prefix found
            feature_mapping[transformed_name] = {
                'original_feature': transformed_name,
                'transformer_type': 'none',
                'is_encoded': False
            }
    
    return feature_mapping


def convert_to_json_serializable(obj):
    """
    Recursively convert numpy arrays and pandas objects to
    JSON-serializable types.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.values.tolist(), obj.columns.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value)
                for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj


def extract_feature_importance(pipeline, feature_names, model_name):
    """
    Extract feature importance from a trained pipeline.

    Parameters:
    -----------
    pipeline : sklearn.pipeline.Pipeline
        Trained pipeline containing feature selector and classifier
    feature_names : list
        Original feature names before pipeline transformations
    model_name : str
        Name of the model for logging purposes

    Returns:
    --------
    pd.DataFrame or None
        DataFrame with columns ['feature_name', 'importance']
        sorted by importance, or None if the model doesn't support
        feature importance
    """
    try:
        # Get the classifier from the pipeline
        classifier = pipeline.named_steps['classifier']

        # Check if the model supports feature importance
        has_feature_importance = hasattr(classifier, 'feature_importances_')
        has_coef = hasattr(classifier, 'coef_')

        if not (has_feature_importance or has_coef):
            return None

        # Get transformed feature names (after preprocessing, before feature selection)
        transformed_feature_names = get_transformed_feature_names(pipeline, feature_names)
        
        logger.info(f"Using {len(transformed_feature_names)} transformed features for importance extraction")

        # Get the feature selector to identify which features were selected
        feature_selector = pipeline.named_steps.get('feature_selector', None)

        if (feature_selector is not None and
                hasattr(feature_selector, 'get_support')):
            # Get the mask of selected features
            selected_features_mask = feature_selector.get_support()
            
            # Ensure mask length matches transformed features length
            if len(selected_features_mask) != len(transformed_feature_names):
                logger.error(
                    f"Mismatch in feature lengths: mask has {len(selected_features_mask)} "
                    f"but transformed features has {len(transformed_feature_names)}"
                )
                return None
            
            selected_feature_names = [
                name for name, selected in
                zip(transformed_feature_names, selected_features_mask)
                if selected
            ]
            logger.info(f"Selected {len(selected_feature_names)} features out of {len(transformed_feature_names)}")
        else:
            # If no feature selector, use all features
            selected_feature_names = transformed_feature_names

        # Extract importance values
        if has_feature_importance:
            # For tree-based models
            importance_values = classifier.feature_importances_
        elif has_coef:
            # For linear models (Logistic Regression, SVM)
            # Use absolute values of coefficients as importance
            coef = classifier.coef_
            if coef.ndim > 1:
                # For multi-class, take the mean of absolute coefficients
                importance_values = np.mean(np.abs(coef), axis=0)
            else:
                importance_values = np.abs(coef)
        else:
            return None

        # Create DataFrame with feature names and importance
        if len(selected_feature_names) != len(importance_values):
            logger.error(
                f"Mismatch: {len(selected_feature_names)} selected features "
                f"but {len(importance_values)} importance values"
            )
            return None

        importance_df = pd.DataFrame({
            'feature_name': selected_feature_names,
            'importance': importance_values
        })

        # Sort by importance (descending)
        importance_df = importance_df.sort_values(
            'importance', ascending=False).reset_index(drop=True)

        return importance_df

    except Exception as e:
        logger.error(f"Error extracting feature importance for "
              f"{model_name}: {str(e)}")
        return None


def extract_selected_features(pipeline, feature_names, model_name):
    """
    Extract the list of features that were selected and used by the model.
    
    This is crucial for deployment - you need to know exactly which features
    to provide when making predictions with a saved model.
    
    Parameters:
    -----------
    pipeline : sklearn.pipeline.Pipeline
        Trained pipeline containing feature selector and classifier
    feature_names : list
        Original feature names before pipeline transformations
    model_name : str
        Name of the model for logging purposes
    
    Returns:
    --------
    list or None
        List of selected feature names in order, or None if extraction fails
    """
    try:
        # Get transformed feature names (after preprocessing, before feature selection)
        transformed_feature_names = get_transformed_feature_names(pipeline, feature_names)
        
        logger.info(f"Extracting selected features from {len(transformed_feature_names)} transformed features")
        
        # Get the feature selector to identify which features were selected
        feature_selector = pipeline.named_steps.get('feature_selector', None)
        
        if (feature_selector is not None and
                hasattr(feature_selector, 'get_support')):
            # Get the mask of selected features
            selected_features_mask = feature_selector.get_support()
            
            # Ensure mask length matches transformed features length
            if len(selected_features_mask) != len(transformed_feature_names):
                logger.error(
                    f"Mismatch in feature lengths: mask has {len(selected_features_mask)} "
                    f"but transformed features has {len(transformed_feature_names)}"
                )
                return None
            
            selected_feature_names = [
                name for name, selected in
                zip(transformed_feature_names, selected_features_mask)
                if selected
            ]
            logger.info(f"Extracted {len(selected_feature_names)} selected features")
        else:
            # If no feature selector, all features are used
            selected_feature_names = transformed_feature_names
            logger.info(f"No feature selector found, using all {len(selected_feature_names)} features")
        
        return selected_feature_names
    
    except Exception as e:
        logger.error(f"Error extracting selected features for "
              f"{model_name}: {str(e)}")
        return None
