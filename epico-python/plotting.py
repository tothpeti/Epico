import os
import pandas as pd
import numpy as np

"""
    Custom files
"""
from plots import create_lineplot_averages_for_all_metrics_and_thresholds, create_boxplot_for_all_metrics_and_thresholds
from plots import create_average_auc_roc_curve, create_min_max_auc_roc_curve, create_3d_plot_for_each_metrics
from plots import create_boxplot_for_col_excluded_datasets, create_lineplot_for_one_metric_col_excluded_datasets
from helper import get_all_datasets_names, get_length_of_test_dataset, read_all_datasets_in_memory, read_all_folders_files_in_memory


def create_plots_for_column_excluded_datasets(file_name, path_to_predictions, path_to_metrics, path_to_diagrams):
    all_pred_dirs = os.listdir(path_to_predictions)
    all_metrics_dirs = os.listdir(path_to_metrics)

    all_pred_datasets = read_all_folders_files_in_memory(all_pred_dirs, path_to_predictions, is_prediction=True)

    # indexes --> 0: accuracy, 1: f1_score, 2: precision, 3: sensitivity, 4: specificity
    all_metrics_datasets = read_all_folders_files_in_memory(all_metrics_dirs, path_to_metrics, is_prediction=False)

    """
    for col_excl_idx, pred_df_list in enumerate(all_pred_datasets):
        create_average_auc_roc_curve(file_name, path_to_predictions, path_to_diagrams, pred_df_list, len(pred_df_list[0]), col_excl_idx)
    """

    thresholds = np.arange(0.0, 1.05, 0.05)

    """
    for col_excl_idx, metric_df_list in enumerate(all_metrics_datasets):
        create_lineplot_averages_for_all_metrics_and_thresholds(file_name, path_to_diagrams, acc_df=metric_df_list[0],
                                                                f1_df=metric_df_list[1], prec_df=metric_df_list[2],
                                                                sens_df=metric_df_list[3], spec_df=metric_df_list[4],
                                                                thresholds=thresholds, col_excl_idx=col_excl_idx)
        """
        #create_3d_plot_for_each_metrics(acc_df=metric_df_list[0], thresholds=thresholds, col_excl_idx=col_excl_idx)

    # create_boxplot_for_col_excluded_datasets(datasets=all_pred_datasets, path_to_diagrams=path_to_diagrams)
    create_lineplot_for_one_metric_col_excluded_datasets(all_metrics_datasets, thresholds, path_to_diagrams)


if __name__ == '__main__':
    path_to_predictions = "D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/Generated_Data_Visualizations/50rounds_10_bern05prob_with_0_to_1_thresholds_20200910/predictions/"
    path_to_predictions_col_excl = "D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/Generated_Data_Visualizations/50rounds_10_bern05prob_with_0_to_1_thresholds_20200910/predictions/column_excluding"
    path_to_metrics = "D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/Generated_Data_Visualizations/50rounds_10_bern05prob_with_0_to_1_thresholds_20200910/metrics/"
    path_to_metrics_col_excl = "D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/Generated_Data_Visualizations/50rounds_10_bern05prob_with_0_to_1_thresholds_20200910/metrics/column_excluding"
    path_to_diagrams = "D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/Generated_Data_Visualizations/50rounds_10_bern05prob_with_0_to_1_thresholds_20200910/diagrams/"
    path_to_diagrams_col_excl = "D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/Generated_Data_Visualizations/50rounds_10_bern05prob_with_0_to_1_thresholds_20200910/diagrams/column_excluding"

    file_name = '50_simrounds_10_bern05prob'

    # Used for single folders
    #datasets = read_all_datasets_in_memory(path_to_predictions)
    #metrics = read_all_datasets_in_memory(path_to_metrics)

    create_plots_for_column_excluded_datasets(file_name, path_to_predictions_col_excl, path_to_metrics_col_excl, path_to_diagrams_col_excl)
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