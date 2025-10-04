from src.ml.data_prep.trainer import Trainer
from src.ml import logger


if __name__ == "__main__":
    logger.info("Starting training process")
    trainer = Trainer()
    logger.info("Trainer initialized successfully")
    trainer.trigger_training('cumulative_2025.10.04_04.05.07.csv', cv=2, n_iter=2, scoring='accuracy', n_jobs=-1)
    logger.info("Training process completed")
