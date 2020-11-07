import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from utils import read_datasets


class Simulation:
    def __init__(self,
                 thresholds: list,
                 threshold_col_names: list,
                 path_to_datasets: str,
                 path_to_metrics: str,
                 path_to_metrics_col_excluding: str,
                 path_to_predictions: str,
                 path_to_predictions_col_excluding: str):
        # Path to files
        self.path_to_datasets = path_to_datasets
        self.path_to_metrics = path_to_metrics
        self.path_to_metrics_col_excluding = path_to_metrics_col_excluding
        self.path_to_predictions = path_to_predictions
        self.path_to_predictions_col_excluding = path_to_predictions_col_excluding

        # Used for preprocessing
        self.pipeline = None

        # Used for model training and testing
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        # Used for prediction
        self.thresholds = thresholds
        self.threshold_col_names = threshold_col_names

        # Main DataFrame
        self.df = pd.DataFrame()

        # Used for storing simulation metrics results
        self.all_accuracy_list = []
        self.all_f1_score_list = []
        self.all_precision_list = []
        self.all_sensitivity_list = []
        self.all_specificity_list = []
        # self.path_to_model_params = ""
        # self.path_to_model_params_col_excluding = ""

    def load_data(self):
        self.df = read_datasets(self.path_to_datasets)
        return self

    def init_pipeline(self, pipeline):
        self.pipeline = pipeline
        return self

    def preprocess_data(self, drop_duplicates=False):
        if drop_duplicates is True:
            self.df.drop_duplicates(ignore_index=True, inplace=True)

    def transform_data(self, features, target, drop_duplicates=False):
        self.x_train, self.y_train, self.x_test, self.y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    def run_model(self, model,
                  features: pd.DataFrame,
                  target: pd.DataFrame,
                  test_size=0.3):

        y_train_idx = self.y_train.index
        y_test_idx = self.y_test.index

        # Preprocess data
        self.transform_data(features=features, target=target)

        # Convert ndarrays to DataFrames
        features_column_names = features.columns
        x_train = pd.DataFrame(data=self.x_train, index=y_train_idx, columns=features_column_names)
        x_test = pd.DataFrame(data=self.y_train, index=y_test_idx, columns=features_column_names)

        # Initialize and train model
        trained_model = model.fit(x_train, self.y_train)

        # Test model
        return test_model(trained_model, x_test, self.y_test)

    def test_mode(self):
        pass

    def show(self):
        print(self.df.head())
