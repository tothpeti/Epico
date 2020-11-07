import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

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
        self.feature_transformer: ColumnTransformer = ColumnTransformer()
        self.feature_cols_idx = []
        self.target_col_idx = None

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

    def init_feature_cols_indexes(self, indexes):
        self.feature_cols_idx = indexes

    def init_target_col_indexes(self, index):
        self.target_col_idx = index

    def init_feature_transformer(self, transformer):
        self.feature_transformer = transformer
        return self

    def apply_feature_transformer(self):
        self.feature_transformer.fit(self.x_train)
        self.x_train = self.feature_transformer.transform(self.x_train)
        self.x_test = self.feature_transformer.transform(self.x_test)
        return self

    def binarize_target(self):
        label_bin = LabelBinarizer().fit(self.y_train)
        self.y_train = label_bin.transform(self.y_train)
        self.y_test = label_bin.transform(self.y_test)
        return self

    def preprocess_data(self, drop_duplicates=False):
        if drop_duplicates is True:
            self.df.drop_duplicates(ignore_index=True, inplace=True)

    def run_model(self, model,
                  features: pd.DataFrame,
                  target: pd.DataFrame,
                  test_size=0.3):
        self.x_train, self.y_train, self.x_test, self.y_test = train_test_split(features,
                                                                                target,
                                                                                test_size=test_size,
                                                                                random_state=42)
        y_train_idx = self.y_train.index
        y_test_idx = self.y_test.index

        self.apply_feature_transformer()\
            .binarize_target()

        # Convert ndarrays to DataFrames
        features_column_names = features.columns
        x_train = pd.DataFrame(data=self.x_train, index=y_train_idx, columns=features_column_names)
        x_test = pd.DataFrame(data=self.y_train, index=y_test_idx, columns=features_column_names)

        # Initialize and train model
        trained_model = model.fit(x_train, self.y_train)

        # Test model
        return self.test_model(trained_model, x_test, self.y_test)

    def test_model(self,
                   trained_model,
                   x_test,
                   y_test):

        # Combine x_test, and y_test into one dataframe
        result_df = pd.DataFrame()
        result_df = pd.concat([result_df, x_test, y_test], axis=1)
        result_df.reset_index(inplace=True, drop=True)

        result_df['y_pred_probs'] = trained_model.predict_proba(x_test)[:, 1]
        result_df['y_pred_probs'] = result_df['y_pred_probs'].round(decimals=3)

        # Create y_predN columns by using threshold values
        for idx in range(len(self.thresholds)):
            result_df[self.threshold_col_names[idx]] = np.nan
            result_df.loc[result_df['y_pred_probs'] >= self.thresholds[idx], self.threshold_col_names[idx]] = 1
            result_df.loc[result_df['y_pred_probs'] < self.thresholds[idx], self.threshold_col_names[idx]] = 0

        return result_df, self.y_test

    def show(self):
        print(self.df.head())
