import traceback
from pathlib import Path

import joblib
import optuna
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config.paths import (
    MODEL_FILE,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    TEST_DATA_FILE,
    TEST_TARGETS_FILE,
    TRAIN_DATA_FILE,
    TRAIN_TARGETS_FILE,
)
from src.custom_exception import CustomException
from src.logger import get_logger

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

    def __init__(self, processed_data_path: Path | str = PROCESSED_DATA_DIR):
        """Initializes the Training class with the processed data path.

        Args:
            processed_data_path (Path | str): Path to the processed data directory.
        """
        self.data_path = processed_data_path
        self.model_dir = MODELS_DIR
        self.model_dir.mkdir(exist_ok=True, parents=True)

        logger.info("Training initialized")

    def load_data(self):
        """Loads the processed data from the specified path.

        Raises:
            CustomException: If there is an error loading the data.
        """
        try:
            X_train = joblib.load(TRAIN_DATA_FILE)
            X_test = joblib.load(TEST_DATA_FILE)
            y_train = joblib.load(TRAIN_TARGETS_FILE)
            y_test = joblib.load(TEST_TARGETS_FILE)

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

    def objective(self, trial):
        try:
            params = {
                "n_estimators": 2000,
                "n_iter_no_change": trial.suggest_int("n_iter_no_change", 10, 50),
                "validation_fraction": 0.1,
                "tol": 1e-4,
                "max_depth": trial.suggest_int("max_depth", 1, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.001, 0.2, log=True
                ),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            }

            # train
            gbc = GradientBoostingClassifier(**params, random_state=42)

            # predict and evaluate
            scores = cross_val_score(
                gbc, self.X_train, self.y_train, cv=5, scoring="f1", n_jobs=-1
            )

            return scores.mean()

        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)[0]
            line_number = tb.lineno
            logger.error("Error tuning hyperparameters - line %d: %s", line_number, e)
            raise CustomException("Error tuning hyperparameters") from e

    def train_model(self):
        """Trains a GradientBoostingClassifier model on the processed data.

        Raises:
            CustomException: If there is an error during model training.
        """
        try:
            study = optuna.create_study(direction="maximize")
            logger.info("Hyperparameter tuning started")
            study.optimize(self.objective, n_trials=10, show_progress_bar=True)

            params = study.best_params

            logger.info("Training started")
            self.model = GradientBoostingClassifier(
                **params,
                random_state=42,
            )

            self.model.fit(self.X_train, self.y_train)

            joblib.dump(self.model, MODEL_FILE)

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
    training = Training(PROCESSED_DATA_DIR)
    training.run()
