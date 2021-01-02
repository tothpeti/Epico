import time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import sys

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler

"""
Custom files
"""
# from processes import run_with_column_excluding, run_without_column_excluding, \
#    run_with_hyperparameter_search_and_column_excluding, run_with_hyperparameter_search_and_without_column_excluding
from utils import get_all_datasets_names, read_all_datasets_in_memory, put_column_excluded_files_into_folders

from simulation import Simulation


if __name__ == '__main__':
    # Home PC
    path_to_datasets = 'D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/DataVisualisations/Scenario5/datasets/'
    path_to_metrics = 'D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/DataVisualisations/Scenario5/tunedROC_randomF_0to1_thresholds/metrics/'
    path_to_metrics_col_excluding = 'D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/DataVisualisations/Scenario5/tunedROC_randomF_0to1_thresholds/metrics/column_excluding/'
    path_to_predictions = 'D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/DataVisualisations/Scenario5/tunedROC_randomF_0to1_thresholds/predictions/'
    path_to_predictions_col_excluding = 'D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/DataVisualisations/Scenario5/tunedROC_randomF_0to1_thresholds/predictions/column_excluding/'
    path_to_model_params = 'D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/DataVisualisations/Scenario5/tunedROC_randomF_0to1_thresholds/best_model_parameters/'
    path_to_model_params_col_excluding = 'D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/DataVisualisations/Scenario5/tunedROC_randomF_0to1_thresholds/best_model_parameters/column_excluding/'
    
    start_time = time.time()

    # datasets_names = get_all_datasets_names(path_to_datasets, True)
    # datasets = read_all_datasets_in_memory(datasets_names, path_to_datasets)

    # Start(inclusive), End(exclusive), Steps
    # 0.0 - 1.0
    thresholds = np.arange(0.0, 1.05, 0.05)

    # Need to round the numbers to 2 decimal places, due to decimal error when try to print out the value
    #   or write into a file
    thresholds = [round(threshold, 2) for threshold in thresholds]
    threshold_col_names = ['y_pred'+str(threshold) for threshold in thresholds]

    # Number of features
    num_of_cols = 10

    sim = Simulation(thresholds=thresholds,
                     threshold_col_names=threshold_col_names,
                     path_to_datasets=path_to_datasets,
                     path_to_metrics=path_to_metrics,
                     path_to_metrics_col_excluding=path_to_metrics_col_excluding,
                     path_to_predictions=path_to_predictions,
                     path_to_predictions_col_excluding=path_to_predictions_col_excluding)
    """
    # Columns to transform
    ord_enc_cols = [1, 3, 5, 6, 7, 8, 9, 13]
    #ord_enc_cols = [1, 3, 5, 6, 7, 13]
    one_hot_cols = [8, 9]
    standard_scaler_cols = [0, 2, 4, 10, 11, 12]

    transformer = ColumnTransformer(
        transformers=[
            ('ord_enc_process', OrdinalEncoder(), ord_enc_cols),
            # ('one_hot_process', OneHotEncoder(drop='first'), one_hot_cols),
            ('standard_scaler_process', StandardScaler(), standard_scaler_cols)
        ],
        remainder="passthrough"
    )
    """
    sim.load_data()

    # Last column is ALWAYS "filename" before that there is "target" column
    features_col_idx = list(range(0, (len(sim.df.columns)-2)))
    target_col_idx = [-2]

    # Rename columns to x_number
    new_cols_names = ["x_"+str(col) for col in features_col_idx]
    new_cols_names.append("y")
    new_cols_names.append("filename")
    sim.df.columns = new_cols_names

    """
    sim.init_feature_transformer(transformer=transformer)\
       .set_feature_cols_indexes(features_col_idx)\
       .set_target_col_indexes(target_col_idx)
    """
    sim.set_feature_cols_indexes(features_col_idx)\
        .set_target_col_indexes(target_col_idx)

    # model = LogisticRegression(n_jobs=-1, random_state=42)

    model = RandomForestClassifier(n_jobs=-1, random_state=42)
    model_params = {
        "n_estimators": [100, 150, 200, 250, 300],
        "max_features": ["sqrt"],
        "max_depth": [None, 5, 10, 15],
        "min_samples_leaf": [1, 2, 5, 10]
    }

    print("Start run without col excl.")
    # sim.run_without_column_excluding(model=model)
    # sim.run_without_column_excluding(model=model, model_params=model_params, use_hyper_opt=True, scoring="roc_auc")
    
    print("Start run with col excl.")
    # sim.run_with_column_excluding(model=model)
    sim.run_with_column_excluding(model=model, model_params=model_params, use_hyper_opt=True, scoring="roc_auc")

    # This step is only needed when we do column excluding runs
    print('Rearranging files finished')
    put_column_excluded_files_into_folders(path_to_metrics_col_excluding)
    put_column_excluded_files_into_folders(path_to_predictions_col_excluding)
    # put_column_excluded_files_into_folders(path_to_model_params_col_excluding)
    print("--- %s seconds ---" % (time.time() - start_time))