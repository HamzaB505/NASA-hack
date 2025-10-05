import numpy as np
import pandas as pd


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

        # Get the feature selector to identify which features were selected
        feature_selector = pipeline.named_steps.get('feature_selector', None)

        if (feature_selector is not None and
                hasattr(feature_selector, 'get_support')):
            # Get the mask of selected features
            selected_features_mask = feature_selector.get_support()
            selected_feature_names = [
                name for name, selected in
                zip(feature_names, selected_features_mask)
                if selected
            ]
        else:
            # If no feature selector, use all features
            selected_feature_names = feature_names

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
            # This shouldn't happen, but as a safety check
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
        print(f"Error extracting feature importance for "
              f"{model_name}: {str(e)}")
        return None
