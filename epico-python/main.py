import glob
import os
import pandas as pd
import numpy as np

"""
Custom files
"""
from metrics import create_metrics, save_metrics, save_prediction_df
from models import run_logistic_reg
from helper import get_all_datasets, put_column_excluded_files_into_folders


if __name__ == '__main__':
    path_to_datasets = "D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/Generated_Data_Visualizations/50rounds_10_bern05prob_with_0_to_1_thresholds_20200910/datasets/"
    path_to_metrics = "D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/Generated_Data_Visualizations/50rounds_10_bern05prob_with_0_to_1_thresholds_20200910/metrics/"
    path_to_metrics_col_excluding = 'D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/Generated_Data_Visualizations/50rounds_10_bern05prob_with_0_to_1_thresholds_20200910/metrics/column_excluding'
    path_to_predictions = "D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/Generated_Data_Visualizations/50rounds_10_bern05prob_with_0_to_1_thresholds_20200910/predictions/"
    path_to_predictions_col_excluding = "D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/Generated_Data_Visualizations/50rounds_10_bern05prob_with_0_to_1_thresholds_20200910/predictions/column_excluding/"

    datasets = get_all_datasets(path_to_datasets)

    """
        Preprocess and model train/test
    """
    # Start(inclusive), End(exclusive), Steps
    # 0.0 - 1.0
    thresholds = np.arange(0.0, 1.05, 0.05)

    # Need to round the numbers to 2 decimal places, due to decimal error when try to print out the value
    #   or write into a file
    thresholds = [round(threshold, 2) for threshold in thresholds]
    threshold_col_names = ['y_pred'+str(threshold) for threshold in thresholds]

    all_accuracy_list = []
    all_f1_score_list = []
    all_precision_list = []
    all_sensitivity_list = []
    all_specificity_list = []

    for col_to_exclude in range(10):
        for dataset in datasets:

            df = pd.read_csv(os.path.join(path_to_datasets, dataset))

            col_name_to_exclude = 'x'+str(col_to_exclude+1)
            features = df.drop(columns=[col_name_to_exclude, 'y'], axis=1)
            target = df['y']
            result_df, y_test = run_logistic_reg(features, target, thresholds, threshold_col_names, test_size=0.3)

            accuracy_list, f1_score_list, precision_list, sensitivity_list, specificity_list = create_metrics(result_df, y_test, threshold_col_names)

            prediction_file_name = dataset.split('.')[0]+'_'+str(col_to_exclude)+'.csv'
            save_prediction_df(result_df, prediction_file_name, path_to_predictions_col_excluding)

            all_accuracy_list.append(accuracy_list)
            all_f1_score_list.append(f1_score_list)
            all_precision_list.append(precision_list)
            all_sensitivity_list.append(sensitivity_list)
            all_specificity_list.append(specificity_list)

        save_metrics(all_accuracy_list, all_f1_score_list,
                     all_precision_list, all_sensitivity_list,
                     all_specificity_list, threshold_col_names,
                     path_to_metrics_col_excluding, str(col_to_exclude))

    put_column_excluded_files_into_folders(path_to_metrics_col_excluding)
    put_column_excluded_files_into_folders(path_to_predictions_col_excluding)
