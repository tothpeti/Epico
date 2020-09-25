import numpy as np

"""
Custom files
"""
from models import run_process_with_column_excluding, run_process_without_column_excluding
from helper import get_all_datasets_names, read_all_datasets_in_memory, put_column_excluded_files_into_folders


if __name__ == '__main__':
    path_to_datasets = "D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/Generated_Data_Visualizations/50rounds_10_bern05prob_with_0_to_1_thresholds_20200910/datasets/"
    path_to_metrics = "D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/Generated_Data_Visualizations/50rounds_10_bern05prob_with_0_to_1_thresholds_20200910/metrics/"
    path_to_metrics_col_excluding = 'D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/Generated_Data_Visualizations/50rounds_10_bern05prob_with_0_to_1_thresholds_20200910/metrics/column_excluding'
    path_to_predictions = "D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/Generated_Data_Visualizations/50rounds_10_bern05prob_with_0_to_1_thresholds_20200910/predictions/"
    path_to_predictions_col_excluding = "D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/Generated_Data_Visualizations/50rounds_10_bern05prob_with_0_to_1_thresholds_20200910/predictions/column_excluding/"

    datasets_names = get_all_datasets_names(path_to_datasets, True)
    datasets = read_all_datasets_in_memory(datasets_names, path_to_datasets)

    # Start(inclusive), End(exclusive), Steps
    # 0.0 - 1.0
    thresholds = np.arange(0.0, 1.05, 0.05)

    # Need to round the numbers to 2 decimal places, due to decimal error when try to print out the value
    #   or write into a file
    thresholds = np.ndarray([round(threshold, 2) for threshold in thresholds])
    threshold_col_names = ['y_pred'+str(threshold) for threshold in thresholds]

    # Number of features
    num_of_cols = 10

    #run_process_without_column_excluding(datasets, datasets_names, thresholds, threshold_col_names,
    #                                     path_to_predictions, path_to_metrics)

    run_process_with_column_excluding(num_of_cols, datasets, datasets_names, thresholds, threshold_col_names,
                                      path_to_predictions_col_excluding, path_to_metrics_col_excluding)

    put_column_excluded_files_into_folders(path_to_metrics_col_excluding)
    put_column_excluded_files_into_folders(path_to_predictions_col_excluding)