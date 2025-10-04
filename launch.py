from src.ml.data_prep.trainer import Trainer
from src.ml.data_prep.utils import convert_to_json_serializable
from src.ml import logger
import json
import os
import logging
from datetime import datetime


if __name__ == "__main__":

    CONFIG = {
        "cv": 5,
        "n_iter": 50,
        "scoring": "accuracy",
        "n_jobs": -1
    }

    logger.info("Starting training process")
    
    # Create timestamped directory for models
    timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    models_dir = "/Users/hamzaboulaala/Documents/github/NASA-hack/models"
    timestamp_models_dir = os.path.join(models_dir, timestamp)
    os.makedirs(timestamp_models_dir, exist_ok=True)
    
    # Set up file logging to save logs in the timestamped models directory
    log_file = os.path.join(timestamp_models_dir, f"training_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Save config to JSON file in the timestamped models directory
    config_file = os.path.join(timestamp_models_dir, f"config_{timestamp}.json")
    with open(config_file, 'w') as f:
        json.dump(CONFIG, f, indent=4)
    
    logger.info(f"Log file created at: {log_file}")
    logger.info(f"Models will be saved to: {timestamp_models_dir}")
    
    # Train the models
    trainer = Trainer()
    logger.info("Trainer initialized successfully")
    results = trainer.trigger_training(
        filename='cumulative_2025.10.04_04.05.07.csv',
        model_save_dir=timestamp_models_dir,
        cv=CONFIG["cv"],
        n_iter=CONFIG["n_iter"],
        scoring=CONFIG["scoring"],
        n_jobs=CONFIG["n_jobs"]
    )

    # Save results to JSON file in the timestamped models directory
    results_file = os.path.join(timestamp_models_dir, f"training_results_{timestamp}.json")

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
    
    # Remove file handler to close the log file properly
    logger.removeHandler(file_handler)
    file_handler.close()
