import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

"""
Custom files
"""
from processes import run_with_column_excluding, run_without_column_excluding, \
    run_with_hyperparameter_search_and_column_excluding, run_with_hyperparameter_search_and_without_column_excluding
from helper import get_all_datasets_names, read_all_datasets_in_memory, put_column_excluded_files_into_folders


if __name__ == '__main__':
    # Home PC

    path_to_datasets = 'D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/DataVisualisations/Scenario2/tunedROC_randomF_0to1_thresholds/datasets/'
    path_to_metrics = 'D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/DataVisualisations/Scenario2/tunedROC_randomF_0to1_thresholds/metrics/'
    path_to_metrics_col_excluding = 'D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/DataVisualisations/Scenario2/tunedROC_randomF_0to1_thresholds/metrics/column_excluding/'
    path_to_predictions = 'D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/DataVisualisations/Scenario2/tunedROC_randomF_0to1_thresholds/predictions/'
    path_to_predictions_col_excluding = 'D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/DataVisualisations/Scenario2/tunedROC_randomF_0to1_thresholds/predictions/column_excluding/'
    path_to_model_params = 'D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/DataVisualisations/Scenario2/tunedROC_randomF_0to1_thresholds/best_model_parameters/'
    path_to_model_params_col_excluding = 'D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/DataVisualisations/Scenario2/tunedROC_randomF_0to1_thresholds/best_model_parameters/column_excluding/'
    
    # Laptop
    """
    path_to_datasets = "C:/Egyetem_es_munka/Egyetem/MSc/Thesis/DataVisualisations/Scenario2/tuned_ROC_random_forest_0to1_threshold/datasets/"
    path_to_metrics = "C:/Egyetem_es_munka/Egyetem/MSc/Thesis/DataVisualisations/Scenario2/tuned_ROC_random_forest_0to1_threshold/metrics/"
    path_to_metrics_col_excluding = 'C:/Egyetem_es_munka/Egyetem/MSc/Thesis/DataVisualisations/Scenario2/tuned_ROC_random_forest_0to1_threshold/metrics/column_excluding'
    path_to_predictions = "C:/Egyetem_es_munka/Egyetem/MSc/Thesis/DataVisualisations/Scenario2/tuned_ROC_random_forest_0to1_threshold/predictions/"
    path_to_predictions_col_excluding = "C:/Egyetem_es_munka/Egyetem/MSc/Thesis/DataVisualisations/Scenario2/tuned_ROC_random_forest_0to1_threshold/predictions/column_excluding/"
    path_to_model_params = 'C:/Egyetem_es_munka/Egyetem/MSc/Thesis/DataVisualisations/Scenario2/tuned_ROC_random_forest_0to1_threshold/best_model_parameters/'
    path_to_model_params_col_excluding = 'C:/Egyetem_es_munka/Egyetem/MSc/Thesis/DataVisualisations/Scenario2/tuned_ROC_random_forest_0to1_threshold/best_model_parameters/column_excluding/'
    """
    start_time = time.time()

    datasets_names = get_all_datasets_names(path_to_datasets, True)
    datasets = read_all_datasets_in_memory(datasets_names, path_to_datasets)

    # Start(inclusive), End(exclusive), Steps
    # 0.0 - 1.0
    thresholds = np.arange(0.0, 1.05, 0.05)

    # Need to round the numbers to 2 decimal places, due to decimal error when try to print out the value
    #   or write into a file
    thresholds = [round(threshold, 2) for threshold in thresholds]
    threshold_col_names = ['y_pred'+str(threshold) for threshold in thresholds]

    # Number of features
    num_of_cols = 10

    # Initialize model
    # model = LogisticRegression(random_state=42, n_jobs=-1)
    model = RandomForestClassifier(random_state=42, n_jobs=-1)

    # RF
    # n_estimators    --> put as high as your CPU can handle!! more is better, but it is very time expensive
    # max_depth       --> maximum number of levels in tree --> None value can be viable!
    # max_features    --> maximum number of features to consider at every split
    # min_samples_leaf --> minimum number of samples required to be a leaf node
    model_params = {
        "n_estimators": [100, 150, 200, 250, 300],
        "max_features": ["sqrt"],
        "max_depth": [None, 5, 10, 15],
        "min_samples_leaf": [1, 2, 5, 10]
    }
    """
    run_with_hyperparameter_search_and_without_column_excluding(model=model,
                                                                model_params=model_params,
                                                                scoring='roc_auc',
                                                                datasets=datasets,
                                                                datasets_names=datasets_names,
                                                                thresholds=thresholds,
                                                                threshold_col_names=threshold_col_names,
                                                                path_to_predictions=path_to_predictions,
                                                                path_to_metrics=path_to_metrics,
                                                                path_to_model_params=path_to_model_params)
    """
    run_with_hyperparameter_search_and_column_excluding(model=model,
                                                        model_params=model_params,
                                                        scoring='roc_auc',
                                                        num_of_cols=num_of_cols,
                                                        datasets=datasets,
                                                        datasets_names=datasets_names,
                                                        thresholds=thresholds,
                                                        threshold_col_names=threshold_col_names,
                                                        path_to_predictions_col_excluding=path_to_predictions_col_excluding,
                                                        path_to_metrics_col_excluding=path_to_metrics_col_excluding,
                                                        path_to_model_params_col_excluding=path_to_model_params_col_excluding)

    """ 
    run_without_column_excluding(model=model,
                                 datasets=datasets,
                                 datasets_names=datasets_names,
                                 thresholds=thresholds,
                                 threshold_col_names=threshold_col_names,
                                 path_to_predictions=path_to_predictions,
                                 path_to_metrics=path_to_metrics)

    print('step 1 finished')
    run_with_column_excluding(model=model,
                              num_of_cols=num_of_cols,
                              datasets=datasets,
                              datasets_names=datasets_names,
                              thresholds=thresholds,
                              threshold_col_names=threshold_col_names,
                              path_to_predictions_col_excluding=path_to_predictions_col_excluding,
                              path_to_metrics_col_excluding=path_to_metrics_col_excluding)
    """
    # This step is only needed when we do column excluding runs
    print('Rearranging files finished')
    put_column_excluded_files_into_folders(path_to_metrics_col_excluding)
    put_column_excluded_files_into_folders(path_to_predictions_col_excluding)
    put_column_excluded_files_into_folders(path_to_model_params_col_excluding)
    print("--- %s seconds ---" % (time.time() - start_time))