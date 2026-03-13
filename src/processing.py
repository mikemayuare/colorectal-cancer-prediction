import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.custom_exception import CustomException
from src.logger import get_logger

logger = get_logger(__name__)


class Processing:
    """Class for data processing including loading, feature selection, and scaling.

    Attributes:
        input_path (Path | str): Path to the input CSV file.
        output_path (Path | str): Directory where processed data and scaler will be saved.
        label_encoder (dict): Dictionary to store LabelEncoder instances for categorical columns.
        scaler (StandardScaler): Scikit-learn StandardScaler instance.
        df (pd.DataFrame): The loaded dataset.
        X (pd.DataFrame | pd.Series | np.ndarray): Feature matrix.
        y (pd.Series | pd.DataFrame | np.ndarray): Target vector.
        feature_names (list): List of selected feature names.
    """

    def __init__(self, input_path: Path | str, output_path: Path | str):
        """Initializes the Processing class with input and output paths.

        Args:
            input_path (Path | str): Path to the input CSV file.
            output_path (Path | str): Directory where processed data and scaler will be saved.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.label_encoder: dict = {}
        self.scaler: StandardScaler = StandardScaler()
        self.df: pd.DataFrame = pd.DataFrame()
        self.X: pd.DataFrame | pd.Series | np.ndarray = pd.DataFrame()
        self.y: pd.Series | pd.DataFrame | np.ndarray = pd.Series()
        self.feature_names: list = []

        Path(self.output_path).mkdir(exist_ok=True, parents=True)
        logger.info("Data processing started")

    def load_data(self):
        """Loads data from the specified input path into a pandas DataFrame.

        Raises:
            CustomException: If there is an error loading the data.
        """
        try:
            self.df = pd.read_csv(self.input_path)
            self.df.columns = self.df.columns.str.lower()
            logger.info("Data loaded successfully")
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)[0]
            line_number = tb.lineno
            logger.error("Error loading data - line %d: %s", line_number, e)
            raise CustomException("Error loading data") from e

    def process_data(self):
        """Preprocesses the data by dropping missing values and unnecessary columns,
        and applying label encoding to categorical features.

        Raises:
            CustomException: If there is an error during data processing.
        """
        try:
            self.df = self.df.dropna(axis=0, how="any").drop(columns=["patient_id"])
            self.X = self.df.drop(columns=["survival_prediction"])
            self.y = self.df["survival_prediction"]

            # label encoding features
            categorical_columns = self.X.select_dtypes(include=["object"]).columns
            for col in categorical_columns:
                le = LabelEncoder()
                self.X[col] = le.fit_transform(self.X[col])
                self.label_encoder[col] = le

            # label encoding target
            le = LabelEncoder()
            self.y = le.fit_transform(self.y)  # type: ignore

            logger.info("Label encoding completed")

        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)[0]
            line_number = tb.lineno
            logger.error("Error processing data - line %d: %s", line_number, e)
            raise CustomException("Error processing data") from e

    def feature_selection(self):
        """Performs feature selection using Chi-Square test to identify the top 5 features.

        Raises:
            CustomException: If there is an error during feature selection.
        """
        try:
            logger.info("Feature selection started")
            X_train, _, y_train, _ = train_test_split(
                self.X,
                self.y,
                test_size=0.2,
                random_state=42,
            )

            X_num = pd.DataFrame(X_train).select_dtypes(include=["int64", "float64"])
            chi2_selector = SelectKBest(chi2, k="all")  # type: ignore
            chi2_selector.fit(X_num, y_train)

            chi2_scores = pd.DataFrame(
                {
                    "feature": X_num.columns,
                    "score": chi2_selector.scores_,
                }
            ).sort_values(by="score", ascending=False)

            self.feature_names = chi2_scores.head(5)["feature"].tolist()
            self.X = self.X[self.feature_names]

            logger.info("Selected features: %s", self.feature_names)
            logger.info("Feature selection completed")

        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)[0]
            line_number = tb.lineno
            logger.error("Error feature selection - line %d: %s", line_number, e)
            raise CustomException("Error feature selection") from e

    def split_and_scale_data(self):
        """Splits the data into training and testing sets and scales them.

        Returns:
            tuple: A tuple containing (X_train, X_test, y_train, y_test).

        Raises:
            CustomException: If there is an error during splitting or scaling.
        """
        try:
            logger.info("Splitting and Scaling started")
            X_train, X_test, y_train, y_test = train_test_split(
                self.X,
                self.y,
                test_size=0.2,
                random_state=42,
                stratify=self.y,
            )
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

            logger.info("Splitting and Scaling completed")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)[0]
            line_number = tb.lineno
            logger.error(
                "Error splitting and scaling data - line %d: %s", line_number, e
            )
            raise CustomException("Error splitting and scaling data") from e

    def save_data_and_scaler(self, X_train, X_test, y_train, y_test):
        """Saves the processed data splits and the scaler to the output directory.

        Args:
            X_train (np.ndarray): Scaled training features.
            X_test (np.ndarray): Scaled testing features.
            y_train (pd.Series | np.ndarray): Training targets.
            y_test (pd.Series | np.ndarray): Testing targets.

        Raises:
            CustomException: If there is an error saving the data or scaler.
        """
        try:
            logger.info("Saving data and scaler started")
            joblib.dump(X_train, self.output_path / Path("X_train.pkl"))
            joblib.dump(X_test, self.output_path / Path("X_test.pkl"))
            joblib.dump(y_train, self.output_path / Path("y_train.pkl"))
            joblib.dump(y_test, self.output_path / Path("y_test.pkl"))

            joblib.dump(self.scaler, self.output_path / Path("scaler.pkl"))

            logger.info("Saving data and scaler completed")

        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)[0]
            line_number = tb.lineno
            logger.error("Error saving data and scaler - line %d: %s", line_number, e)
            logger.error("Error saving data and scaler: %s", e)
            raise CustomException("Error saving data and scaler") from e

    def run(self):
        """Executes the full data processing pipeline."""
        self.load_data()
        self.process_data()
        self.feature_selection()
        X_train, X_test, y_train, y_test = self.split_and_scale_data()
        self.save_data_and_scaler(X_train, X_test, y_train, y_test)

        logger.info("Data processing completed")


if __name__ == "__main__":
    input_path = Path("artifacts/raw_data/data.csv")
    output_path = Path("artifacts/processed_data")

    processing = Processing(input_path, output_path)
    processing.run()
