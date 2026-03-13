import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.custom_exception import CustomException
from src.logger import get_logger
from src.processing import Processing

logger = get_logger(__name__)


class Training:
    """Class for training a model on the processed data.

    Attributes:
        processing (Processing): Instance of the Processing class.
        X_train (np.ndarray): Scaled training features.
        X_test (np.ndarray): Scaled testing features.
        y_train (pd.Series | np.ndarray): Training targets.
        y_test (pd.Series | np.ndarray): Testing targets.
        model (RandomForestClassifier): Scikit-learn RandomForestClassifier instance.
    """

    def __init__(
        self, processed_data_path: Path | str = Path("artifacts/processed_data")
    ):
        """Initializes the Training class with the processed data path.

        Args:
            processed_data_path (Path | str): Path to the processed data directory.
        """
        self.data_path = processed_data_path
        self.model_dir = Path("artifacts/models")
        self.model_dir.mkdir(exist_ok=True, parents=True)

        logger.info("Training started")

    def load_data(self):
        """Loads the processed data from the specified path.

        Raises:
            CustomException: If there is an error loading the data.
        """
        try:
            X_train = joblib.load(self.data_path / Path("X_train.pkl"))
            X_test = joblib.load(self.data_path / Path("X_test.pkl"))
            y_train = joblib.load(self.data_path / Path("y_train.pkl"))
            y_test = joblib.load(self.data_path / Path("y_test.pkl"))

            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test

            logger.info("Data loaded successfully")

        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)[0]
            line_number = tb.lineno
            logger.error("Error loading data - line %d: %s", line_number, e)
            raise CustomException("Error loading data") from e

    def train_model(self):
        """Trains a GradientBoostingClassifier model on the processed data.

        Raises:
            CustomException: If there is an error during model training.
        """
        try:
            logger.info("Training started")
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )
            self.model.fit(self.X_train, self.y_train)

            joblib.dump(self.model, self.model_dir / Path("model.pkl"))

            logger.info("Model trained and saved successfully")
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)[0]
            line_number = tb.lineno
            logger.error("Error training model - line %d: %s", line_number, e)
            raise CustomException("Error training model") from e

    def evaluate_model(self):
        """Evaluates the trained model on the testing data.

        Raises:
            CustomException: If there is an error during model evaluation.
        """
        try:
            logger.info("Model evaluation started")
            y_pred = self.model.predict(self.X_test)
            y_proba = self.model.predict_proba(self.X_test)

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(
                self.y_test, y_proba[:, 1] if y_proba.shape[1] == 2 else None
            )

            logger.info("Model evaluation completed")
            logger.info("Accuracy: %.2f", accuracy)
            logger.info("Precision: %.2f", precision)
            logger.info("Recall: %.2f", recall)
            logger.info("F1 Score: %.2f", f1)
            logger.info("ROC AUC: %.2f", roc_auc)
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)[0]
            line_number = tb.lineno
            logger.error("Error evaluating model - line %d: %s", line_number, e)
            raise CustomException("Error evaluating model") from e

    def run(self):
        """Executes the full model training and evaluation pipeline."""
        self.load_data()
        self.train_model()
        self.evaluate_model()

        logger.info("Training completed")


if __name__ == "__main__":
    training = Training(Path("artifacts/processed_data"))
    training.run()
