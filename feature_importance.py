import json
import os
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from src.ml import logger


def load_model_and_get_feature_importance(model_path, feature_names=None):
    """
    Load XGBoost model and extract feature importance
    
    Args:
        model_path (str): Path to the XGBoost pickle file
        feature_names (list): List of feature names
    
    Returns:
        dict: Feature importance scores
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Navigate through the pipeline structure to find the actual XGBoost model
        # The model might be wrapped in a pipeline or calibration wrapper
        actual_model = model["trained_pipeline"].named_steps["classifier"]
        logger.info(f"Actual model type: {actual_model}")
        
        # If it's a dictionary (like from cross-validation results)
        if isinstance(model, dict):
            # Look for common keys that might contain the model
            if 'best_estimator_' in model:
                actual_model = model['best_estimator_']
            elif 'estimator' in model:
                actual_model = model['estimator']
            else:
                # Try to find any sklearn estimator in the dict values
                for key, value in model.items():
                    if hasattr(value, 'feature_importances_') or hasattr(value, 'estimator'):
                        actual_model = value
                        break
        
        # If it's a calibrated classifier, get the base estimator
        if hasattr(actual_model, 'calibrated_classifiers_'):
            # CalibratedClassifierCV has calibrated_classifiers_ list
            actual_model = actual_model.calibrated_classifiers_[0].estimator
        elif hasattr(actual_model, 'base_estimator'):
            # Some wrappers use base_estimator
            actual_model = actual_model.base_estimator
        elif hasattr(actual_model, 'estimator'):
            # Some wrappers use estimator
            actual_model = actual_model.estimator
        
        # If it's a pipeline, get the final estimator
        if hasattr(actual_model, 'steps'):
            # It's a Pipeline, get the last step
            actual_model = actual_model.steps[-1][1]
        elif hasattr(actual_model, 'named_steps'):
            # Alternative pipeline access
            # Try to find XGBoost classifier in named steps
            for step_name, step_estimator in actual_model.named_steps.items():
                if hasattr(step_estimator, 'feature_importances_'):
                    actual_model = step_estimator
                    break
        
        # Final check - if still wrapped, try to unwrap further
        while hasattr(actual_model, 'estimator') and not hasattr(actual_model, 'feature_importances_'):
            actual_model = actual_model.estimator
        
        # Get feature importance from the actual XGBoost model
        if hasattr(actual_model, 'feature_importances_'):
            importance_scores = actual_model.feature_importances_
        else:
            logger.error(f"Could not find feature_importances_ in model structure. "
                        f"Model type: {type(actual_model)}")
            logger.error(f"Available attributes: {dir(actual_model)}")
            return None
        
        if feature_names is None:
            feature_names = [f'feature_{i}' 
                            for i in range(len(importance_scores))]
        
        # Create feature importance dictionary
        feature_importance = dict(zip(feature_names, importance_scores))
        
        # Sort by importance (descending)
        feature_importance = dict(sorted(feature_importance.items(),
                                        key=lambda x: x[1], reverse=True))
        
        return feature_importance
        
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}")
        return None


def create_feature_importance_plot(feature_importance, datatype, save_dir,
                                  top_n=20):
    """
    Create and save feature importance plot
    
    Args:
        feature_importance (dict): Feature importance scores
        datatype (str): Dataset type (KEPLER, TESS, K2)
        save_dir (str): Directory to save the plot
        top_n (int): Number of top features to display
    """
    if not feature_importance:
        logger.warning("No feature importance data to plot")
        return
    
    # Get top N features
    top_features = dict(list(feature_importance.items())[:top_n])
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    features = list(top_features.keys())
    scores = list(top_features.values())
    
    # Create horizontal bar plot
    bars = plt.barh(range(len(features)), scores, color='skyblue', alpha=0.8)
    
    # Customize the plot
    plt.yticks(range(len(features)), features)
    plt.xlabel('Feature Importance Score')
    plt.title(f'Top {top_n} Feature Importance - XGBoost ({datatype})')
    plt.gca().invert_yaxis()  # Highest importance at top
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        plt.text(score + max(scores) * 0.01, i, f'{score:.4f}',
                 va='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f'feature_importance_xgboost_{datatype.lower()}.png'
    plot_path = os.path.join(save_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Feature importance plot saved to: {plot_path}")
    return plot_path


def save_feature_importance_json(feature_importance, datatype, save_dir):
    """
    Save feature importance to JSON file
    
    Args:
        feature_importance (dict): Feature importance scores
        datatype (str): Dataset type
        save_dir (str): Directory to save the JSON file
    """
    if not feature_importance:
        logger.warning("No feature importance data to save")
        return
    
    # Prepare data for JSON
    importance_data = {
        "datatype": datatype,
        "model": "XGBoost",
        "timestamp": datetime.now().isoformat(),
        "feature_importance": feature_importance,
        "top_10_features": dict(list(feature_importance.items())[:10])
    }
    
    # Save to JSON
    json_filename = f'feature_importance_xgboost_{datatype.lower()}.json'
    json_path = os.path.join(save_dir, json_filename)
    
    with open(json_path, 'w') as f:
        json.dump(importance_data, f, indent=4)
    
    logger.info(f"Feature importance JSON saved to: {json_path}")
    return json_path


def process_feature_importance_for_datatype(models_base_dir, datatype_str,
                                           feature_names=None):
    """
    Process feature importance for a specific datatype
    
    Args:
        models_base_dir (str): Base directory containing model folders
        datatype_str (str): String representation of datatype 
                            (e.g., "KEPLER", "TESS")
        feature_names (list): List of feature names
    """
    logger.info(f"Processing feature importance for {datatype_str}")
    
    # Use specific model path based on telescope
    if datatype_str == "TESS":
        model_path = ("/Users/hamzaboulaala/Documents/github/NASA-hack/"
                      "models/2025.10.05_16.07.33/TESS/XGBoost_model.pkl")
        save_dir = ("/Users/hamzaboulaala/Documents/github/NASA-hack/"
                    "models/2025.10.05_16.07.33/TESS")
    elif datatype_str == "KEPLER":
        model_path = ("/Users/hamzaboulaala/Documents/github/NASA-hack/"
                      "models/2025.10.05_16.07.33/KEPLER/XGBoost_model.pkl")
        save_dir = ("/Users/hamzaboulaala/Documents/github/NASA-hack/"
                    "models/2025.10.05_16.07.33/KEPLER")
    else:
        logger.error(f"Unsupported datatype: {datatype_str}")
        return
    
    # Check if model file exists
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return
    
    logger.info(f"Using model path: {model_path}")
    
    # Load model and get feature importance
    feature_importance = load_model_and_get_feature_importance(model_path,
                                                              feature_names)
    
    if feature_importance:
        # Create and save plot
        create_feature_importance_plot(feature_importance, datatype_str,
                                      save_dir)
        
        # Save to JSON
        save_feature_importance_json(feature_importance, datatype_str,
                                    save_dir)
        
        logger.info(f"Feature importance analysis completed for "
                    f"{datatype_str}")
        logger.info(f"Top 5 features: "
                    f"{list(feature_importance.keys())[:5]}")
    else:
        logger.error(f"Failed to extract feature importance for "
                     f"{datatype_str}")


if __name__ == "__main__":
    # Configuration
    models_dir = "/Users/hamzaboulaala/Documents/github/NASA-hack/models"
    
    # Define feature names (you may need to adjust these based on your 
    # actual features)
    # These should match the features used during training
    feature_names = None  # Will use default feature names if None
    
    # Process feature importance for each datatype
    datatypes = ["KEPLER", "TESS"]  # Add "K2" if needed
    
    for datatype_str in datatypes:
        try:
            process_feature_importance_for_datatype(models_dir, datatype_str,
                                                    feature_names)
        except Exception as e:
            logger.error(f"Error processing {datatype_str}: {str(e)}")
    
    logger.info("Feature importance analysis completed for all datatypes")
