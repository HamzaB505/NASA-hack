from src.ml.data_prep.trainer import Trainer
from src.ml import logger
import json
import os
from datetime import datetime
import numpy as np
import pandas as pd


def convert_to_json_serializable(obj):
    """
    Recursively convert numpy arrays and pandas objects to JSON-serializable types.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.values.tolist(), obj.columns.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj


if __name__ == "__main__":
    logger.info("Starting training process")
    trainer = Trainer()
    logger.info("Trainer initialized successfully")
    results = trainer.trigger_training(
        'cumulative_2025.10.04_04.05.07.csv',
        cv=2,
        n_iter=2,
        scoring='accuracy',
        n_jobs=-1
    )

    # Create results directory
    results_dir = "/Users/hamzaboulaala/Documents/github/NASA-hack/results"
    os.makedirs(results_dir, exist_ok=True)
    # Save results to JSON file with timestamp
    timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    timestamp_dir = os.path.join(results_dir, timestamp)
    os.makedirs(timestamp_dir, exist_ok=True)
    results_file = os.path.join(timestamp_dir, f"training_results_{timestamp}.json")

    # Convert results to JSON-serializable format
    results_serializable = convert_to_json_serializable(results)
    if isinstance(results_serializable, tuple):
        results_metrics = results_serializable[0]
        results_labels = results_serializable[1]

        results_serializable = {
            "metrics": results_metrics,
            "labels": results_labels
        }

        with open(results_file, 'w') as f:
            json.dump(results_serializable, f, indent=4)
    else:
        with open(results_file, 'w') as f:
            json.dump(results_serializable, f, indent=4)

    logger.info(f"Training results saved to: {results_file}")
    logger.info("Training process completed")
