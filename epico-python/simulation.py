import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelBinarizer

from metrics import create_metrics, save_prediction_df, save_metrics
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
        self.feature_transformer = None
        self.feature_cols_idx = []
        self.target_col_idx = 0

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
        self.file_names = []

        # Used for storing simulation metrics results
        self.all_accuracy_list = []
        self.all_f1_score_list = []
        self.all_precision_list = []
        self.all_sensitivity_list = []
        self.all_specificity_list = []
        # self.path_to_model_params = ""
        # self.path_to_model_params_col_excluding = ""

    ####################################
    #     Start of methods section
    ####################################
    def load_data(self):
        self.df = read_datasets(self.path_to_datasets)
        self.file_names = list(self.df["filename"].unique())

    def set_feature_cols_indexes(self, indexes):
        self.feature_cols_idx = indexes
        return self

    def set_target_col_indexes(self, index):
        self.target_col_idx = index
        return self

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

    def preprocess_data(self, drop_duplicates=False):
        if drop_duplicates is True:
            self.df.drop_duplicates(ignore_index=True, inplace=True)

    def run_model(self, model,
                  features: pd.DataFrame,
                  target: pd.DataFrame,
                  test_size=0.3):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(features,
                                                                                target,
                                                                                test_size=test_size,
                                                                                random_state=42)
        y_train_idx = self.y_train.index
        y_test_idx = self.y_test.index

        # Transform data into new form
        self.apply_feature_transformer()\
            .binarize_target()

        # Convert ndarrays to DataFrames
        features_column_names = features.columns
        print(features_column_names)
        print(self.x_train.columns)
        self.x_train = pd.DataFrame(data=self.x_train, index=y_train_idx, columns=features_column_names)
        self.x_test = pd.DataFrame(data=self.x_test, index=y_test_idx, columns=features_column_names)

        # Initialize and train model
        trained_model = model.fit(self.x_train, self.y_train)

        # Test model
        return self.test_model(trained_model, self.x_test, self.y_test)

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

        return result_df

    def run_without_column_excluding(self,
                                     model,
                                     model_params=None,
                                     use_hyper_opt=False,
                                     scoring=None):
        if model_params is None:
            model_params = {}

        for filename in self.file_names:
            tmp_df = self.df.loc[self.df["filename"] == filename]
            features = tmp_df.iloc[:, self.feature_cols_idx]
            target = tmp_df.iloc[:, self.target_col_idx]

            result_df = pd.DataFrame()
            if use_hyper_opt is False:
                result_df = self.run_model(model=model,
                                           features=features,
                                           target=target)
            else:
                clf = RandomizedSearchCV(model,
                                         model_params,
                                         cv=5, n_iter=50,
                                         refit=True,
                                         verbose=0, n_jobs=-1,
                                         scoring=scoring)

                result_df = self.run_model(model=clf,
                                           features=features,
                                           target=target)

            accuracy_list, f1_score_list, precision_list, sensitivity_list, specificity_list = create_metrics(
                result_df,
                self.y_test,
                self.threshold_col_names)

            self.all_accuracy_list.append(accuracy_list)
            self.all_f1_score_list.append(f1_score_list)
            self.all_precision_list.append(precision_list)
            self.all_sensitivity_list.append(sensitivity_list)
            self.all_specificity_list.append(specificity_list)

            save_prediction_df(result_df, filename, self.path_to_predictions)
            print("-- Finished with " + filename)

        save_metrics(self.all_accuracy_list, self.all_f1_score_list,
                     self.all_precision_list, self.all_sensitivity_list,
                     self.all_specificity_list, self.threshold_col_names,
                     self.path_to_metrics)

    def run_with_column_excluding(self,
                                  model,
                                  model_params=None,
                                  use_hyper_opt=False,
                                  scoring=None):
        if model_params is None:
            model_params = {}

        for filename in self.file_names:

            tmp_df = self.df.loc[self.df["filename"] == filename]
            curr_col = 0
            for col_to_exclude in self.feature_cols_idx:
                # used for saving metrics
                curr_col = col_to_exclude

                tmp_feature_cols_idx = self.feature_cols_idx

                features = tmp_df.iloc[:, tmp_feature_cols_idx.remove(col_to_exclude)]
                target = tmp_df.iloc[:, self.target_col_idx]

                result_df = pd.DataFrame()
                if use_hyper_opt is False:
                    result_df = self.run_model(model=model,
                                               features=features,
                                               target=target)
                else:
                    clf = RandomizedSearchCV(model,
                                             model_params,
                                             cv=5, n_iter=50,
                                             refit=True,
                                             verbose=0, n_jobs=-1,
                                             scoring=scoring)

                    result_df = self.run_model(model=clf,
                                               features=features,
                                               target=target)

                accuracy_list, f1_score_list, precision_list, sensitivity_list, specificity_list = create_metrics(
                    result_df,
                    self.y_test,
                    self.threshold_col_names)

                prediction_file_name = filename.split('.')[0]+'_'+str(col_to_exclude)+'.csv'
                save_prediction_df(result_df, prediction_file_name, self.path_to_predictions_col_excluding)

                self.all_accuracy_list.append(accuracy_list)
                self.all_f1_score_list.append(f1_score_list)
                self.all_precision_list.append(precision_list)
                self.all_sensitivity_list.append(sensitivity_list)
                self.all_specificity_list.append(specificity_list)

                print('Finished with '+filename+' dataset, column excluded: '+str(col_to_exclude))

            save_metrics(self.all_accuracy_list, self.all_f1_score_list,
                         self.all_precision_list, self.all_sensitivity_list,
                         self.all_specificity_list, self.threshold_col_names,
                         self.path_to_metrics_col_excluding, str(curr_col))

    def show(self):
        print(self.df.head())
