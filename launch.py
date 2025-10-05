from src.ml.data_prep.trainer import Trainer
from src.ml import logger
import json
import os
import logging
from datetime import datetime
from src.ml.data_prep.preprocessing import DATATYPE


if __name__ == "__main__":
    data_dir = "/Users/hamzaboulaala/Documents/github/NASA-hack/data"
    # Create timestamped directory for models
    timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    models_dir = "/Users/hamzaboulaala/Documents/github/NASA-hack/models"
    timestamp_models_dir = os.path.join(models_dir, timestamp)
    os.makedirs(timestamp_models_dir, exist_ok=True)
    
    data_files = {
        "KEPLER": "cumulative_2025.10.04_04.05.07.csv",
        "TESS": "TOI_2025.10.05_02.32.22.csv",
        # "K2": "k2pandc_2025.10.05_02.32.26.csv"
    }
    for datatype, filename in data_files.items():
        if datatype == "KEPLER":
            datatype = DATATYPE.KEPLER
        elif datatype == "TESS":
            datatype = DATATYPE.TESS
        elif datatype == "K2":
            datatype = DATATYPE.K2

        CONFIG = {
            "datatype": datatype,  # Options: "KEPLER", "TESS", "K2"
            "cv": 2,
            "n_iter": 10,
            "n_points": 5,
            "scoring": "accuracy",
            "n_jobs": -1
        }

        logger.info("Starting training process")
        logger.info(f"Training with datatype: {datatype}")
        logger.info(f"Training with filename: {filename}")

        # Create dataset type folder within the timestamped directory
        dataset_type_dir = os.path.join(timestamp_models_dir, CONFIG["datatype"].value)
        os.makedirs(dataset_type_dir, exist_ok=True)

        # Set up file logging to save logs in the dataset type directory
        log_file = os.path.join(dataset_type_dir, f"training_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Save config to JSON file in the dataset type directory

        logger.info(f"Log file created at: {log_file}")
        logger.info(f"Models will be saved to: {dataset_type_dir}")

        # Train the models
        trainer = Trainer()
        logger.info("Trainer initialized successfully")
        results = trainer.trigger_training(
            filename=filename,
            data_dir=data_dir,
            datatype=CONFIG["datatype"],
            model_save_dir=dataset_type_dir,
            cv=CONFIG["cv"],
            n_iter=CONFIG["n_iter"],
            n_points=CONFIG["n_points"],
            scoring=CONFIG["scoring"],
            n_jobs=CONFIG["n_jobs"]
        )

        # Save results to JSON file in the dataset type directory
        results_file = os.path.join(
            dataset_type_dir, f"test_results_{timestamp}.json")

        # Convert results to the desired format
        formatted_results = {"metrics": {}}

        for model_name, model_data in results.items():
            # Get the metrics DataFrame
            metrics_df = model_data["metrics"]

            # Convert to dict with metrics_names and values
            # Extract columns (skip the 'Model' column)
            columns = [col for col in metrics_df.columns if col != 'Model']

            # Extract values for the current model (first row)
            values = metrics_df.iloc[0][columns].tolist()

            formatted_results["metrics"][model_name] = {
                "metrics_names": columns,
                "values": values,
                "confusion_matrix": model_data.get("confusion_matrix", None)
            }
        with open(results_file, 'w') as f:
            json.dump(formatted_results, f, indent=4)

        logger.info(f"Training results saved to: {results_file}")
        logger.info("Training process completed")

        CONFIG["n_train"] = trainer.n_train
        CONFIG["n_test"] = trainer.n_test
        CONFIG["n_features"] = trainer.n_features
        CONFIG["n_classes"] = trainer.n_classes
        CONFIG["datatype"] = CONFIG["datatype"].value

        config_file = os.path.join(dataset_type_dir, "experiment_config.json")
        with open(config_file, 'w') as f:
            json.dump(CONFIG, f, indent=4)
        logger.info(f"Config saved to: {config_file}")

        # Remove file handler to close the log file properly
        logger.removeHandler(file_handler)
        file_handler.close()
