import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

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
        self.path_to_datasets = path_to_datasets
        self.path_to_metrics = path_to_metrics
        self.path_to_metrics_col_excluding = path_to_metrics_col_excluding
        self.path_to_predictions = path_to_predictions
        self.path_to_predictions_col_excluding = path_to_predictions_col_excluding
        self.thresholds = thresholds
        self.threshold_col_names = threshold_col_names
        self.df = None
        # self.path_to_model_params = ""
        # self.path_to_model_params_col_excluding = ""

    def load_data(self):
        self.df = read_datasets(self.path_to_datasets)
        return self

    def preprocess_data(self):
        # Return preprocessed and train_test_split datasets
        pass

    def show(self):
        print(self.df.head())
