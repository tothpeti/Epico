import os
import pandas as pd
import numpy as np

"""
    Custom files
"""
from plots import create_lineplot_averages_for_all_metrics_and_thresholds, create_boxplot_for_all_metrics_and_thresholds
from plots import create_average_auc_roc_curve, create_min_max_auc_roc_curve
from helper import get_all_datasets_names, get_length_of_test_dataset, read_all_datasets_in_memory, read_all_folders_files_in_memory


def create_plots_for_column_excluded_datasets(file_name, path_to_predictions, path_to_metrics, path_to_diagrams):
    all_pred_dirs = os.listdir(path_to_predictions)
    all_metrics_dirs = os.listdir(path_to_metrics)

    all_pred_datasets = read_all_folders_files_in_memory(all_pred_dirs, path_to_predictions)

    # indexes --> 0: accuracy, 1: f1_score, 2: precision, 3: sensitivity, 4: specificity
    all_metrics_datasets = read_all_folders_files_in_memory(all_metrics_dirs, path_to_metrics)

    for idx in range(len(all_pred_datasets)):
        pass


if __name__ == '__main__':
    path_to_predictions = "D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/Generated_Data_Visualizations/50rounds_10_bern05prob_with_0_to_1_thresholds_20200910/predictions/"
    path_to_predictions_col_excl = "D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/Generated_Data_Visualizations/50rounds_10_bern05prob_with_0_to_1_thresholds_20200910/predictions/column_excluding"
    path_to_metrics = "D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/Generated_Data_Visualizations/50rounds_10_bern05prob_with_0_to_1_thresholds_20200910/metrics/"
    path_to_metrics_col_excl = "D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/Generated_Data_Visualizations/50rounds_10_bern05prob_with_0_to_1_thresholds_20200910/metrics/column_excluding"
    path_to_diagrams = "D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/Generated_Data_Visualizations/50rounds_10_bern05prob_with_0_to_1_thresholds_20200910/diagrams/"
    path_to_diagrams_col_excl = "D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/Generated_Data_Visualizations/50rounds_10_bern05prob_with_0_to_1_thresholds_20200910/diagrams/column_excluding"

    file_name = '50_simrounds_10_bern05prob'

    #datasets = get_all_datasets_names(path_to_predictions)
    #metrics = get_all_datasets_names(path_to_metrics)

    create_plots_for_column_excluded_datasets(path_to_predictions_col_excl, path_to_metrics_col_excl)
    """
    # Open previously saved metrics
    accuracy_df = pd.read_csv(os.path.join(path_to_metrics, metrics[0]))
    f1score_df = pd.read_csv(os.path.join(path_to_metrics, metrics[1]))
    precision_df = pd.read_csv(os.path.join(path_to_metrics, metrics[2]))
    sensitivity_df = pd.read_csv(os.path.join(path_to_metrics, metrics[3]))
    specificity_df = pd.read_csv(os.path.join(path_to_metrics, metrics[4]))

    #create_boxplot_for_all_metrics_and_thresholds(file_name, path_to_diagrams, accuracy_df, precision_df, f1score_df,
    #                                              specificity_df, sensitivity_df)

    thresholds = np.arange(0.0, 1.05, 0.05)
    create_lineplot_averages_for_all_metrics_and_thresholds(file_name, path_to_diagrams_without_excl, accuracy_df, precision_df, f1score_df,
                                                             specificity_df, sensitivity_df, thresholds)

    #create_min_max_auc_roc_curve(file_name, path_to_predictions, path_to_diagrams, datasets)

    len_of_test_dataset = get_length_of_test_dataset(path_to_predictions, datasets[0])
    create_average_auc_roc_curve(file_name, path_to_predictions, path_to_diagrams_without_excl, datasets, len_of_test_dataset)
    """