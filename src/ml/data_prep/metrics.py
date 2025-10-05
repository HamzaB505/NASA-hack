# Metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
import os
from src.ml import logger


def compute_and_show_confusion_matrix(
        y_true: pd.Series,
        y_pred: pd.Series,
        model_name: str = "Model",
        normalize: bool = None,
        display_labels: list[str] = None,
        save_dir: str = None):
    """
    Compute and display confusion matrix with nice formatting.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    model_name : str, default="Model"
        Name of the model for display
    normalize : {'true', 'pred', 'all'}, default=None
        Normalizes confusion matrix over the true (rows), predicted (columns)
        conditions or all the population. If None, confusion matrix will not be normalized.
    display_labels : array-like of shape (n_classes,), default=None
        Target names used for plotting. By default, labels will be used if it is defined,
        otherwise the unique labels of y_true and y_pred will be used.
    save_dir : str, default=None
        Directory to save the confusion matrix heatmap. If None, uses default models directory.
    
    Returns:
    --------
    pd.DataFrame
        Confusion matrix as a DataFrame with proper labels
    """

    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    logger.info(f"Confusion matrix: {y_true.shape} vs {y_pred.shape}")
    logger.info(f"y_pred:{y_pred}")
    logger.info(f"y_true:{y_true}")
    
    # Get unique labels
    if display_labels is None:
        # Convert both to numpy arrays and ensure same dtype
        y_true_array = np.asarray(y_true)
        y_pred_array = np.asarray(y_pred)
        labels = sorted(list(set(y_true_array) | set(y_pred_array)))
    else:
        labels = display_labels
    
    # Create DataFrame for better visualization
    cm_df = pd.DataFrame(
                    cm,
                    index=[f'True {label}' for label in labels],
                    columns=[f'Pred {label}' for label in labels])
    
    logger.info(f"\nConfusion Matrix for {model_name}")
    logger.info("=" * (len(model_name) + 20))
    logger.info(cm_df.round(3))
    
    # Create and save confusion matrix heatmap
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', values_format='.3f' if normalize else 'd')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    
    # Save the heatmap
    if save_dir is None:
        save_dir = "/Users/hamzaboulaala/Documents/github/NASA-hack/models"
    os.makedirs(save_dir, exist_ok=True)
    clean_name = model_name.replace(' ', '_').replace('/', '_')
    heatmap_path = os.path.join(save_dir, f'{clean_name}_confusion_matrix.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved confusion matrix heatmap to {heatmap_path}")
    
    return cm_df


def get_classification_metrics(y_true, y_pred, y_pred_proba=None, model_name="Model"):
    """
    Calculate comprehensive classification metrics and return them in a nice table format.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like, optional
        Predicted probabilities (for ROC AUC and PR AUC)
    model_name : str, default="Model"
        Name of the model for the table
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing all metrics in a nice table format
    """
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Initialize metrics dictionary
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    
    # Add probability-based metrics if available
    if y_pred_proba is not None:
        try:
            # Ensure y_pred_proba is a numpy array
            y_pred_proba = np.asarray(y_pred_proba)
            
            # For binary classification
            if len(np.unique(y_true)) == 2:
                roc_auc = roc_auc_score(
                    y_true,
                    y_pred_proba[:, 1]
                    if y_pred_proba.ndim > 1 else y_pred_proba)
                pr_auc = average_precision_score(
                    y_true,
                    y_pred_proba[:, 1]
                    if y_pred_proba.ndim > 1 else y_pred_proba)
                metrics['ROC AUC'] = roc_auc
                metrics['PR AUC'] = pr_auc
            else:
                # For multiclass
                roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
                metrics['ROC AUC'] = roc_auc
        except Exception as e:
            # If probability metrics fail, log the error and skip them
            logger.warning(f"Failed to compute AUC metrics for {model_name}: {str(e)}")
            pass
    
    # Create DataFrame
    metrics_df = pd.DataFrame([metrics])
    
    # Round numerical values for better display
    numerical_cols = metrics_df.select_dtypes(include=[np.number]).columns
    metrics_df[numerical_cols] = metrics_df[numerical_cols].round(4)
    
    return metrics_df


def compare_models_metrics(models_results):
    """
    Compare multiple models' metrics in a single table.
    
    Parameters:
    -----------
    models_results : dict
        Dictionary where keys are model names and values are tuples of (y_true, y_pred, y_pred_proba)
        Example: {'LogisticRegression': (y_true, y_pred, y_pred_proba), ...}
    
    Returns:
    --------
    pd.DataFrame
        DataFrame comparing all models' metrics
    """
    
    all_metrics = []
    
    for model_name, results in models_results.items():
        y_true = results["y_test"]
        y_pred = results["y_pred"]
        y_pred_proba = results["y_pred_proba"]
        model_metrics = get_classification_metrics(
            y_true, y_pred, y_pred_proba, model_name)
        all_metrics.append(model_metrics)
    
    # Combine all metrics
    comparison_df = pd.concat(all_metrics, ignore_index=True)
    
    # Sort by F1-Score (or any other metric you prefer)
    comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
    
    return comparison_df

