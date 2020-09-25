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


def create_plots_for_without_column_excluded_datasets(file_name: str,
                                                      path_to_predictions: str,
                                                      path_to_metrics: str,
                                                      path_to_diagrams: str) -> None:

    datasets_names = get_all_datasets_names(path_to_predictions, True)
    metrics_names = get_all_datasets_names(path_to_metrics, False)

    datasets = read_all_datasets_in_memory(datasets_names, path_to_predictions)

    # indexes --> 0: accuracy, 1: f1_score, 2: precision, 3: sensitivity, 4: specificity
    metrics = read_all_datasets_in_memory(metrics_names, path_to_metrics)

    create_boxplot_for_all_metrics_and_thresholds(file_name,
                                                  path_to_diagrams,
                                                  acc_df=metrics[0],
                                                  f1_df=metrics[1],
                                                  prec_df=metrics[2],
                                                  sens_df=metrics[3],
                                                  spec_df=metrics[4])

    thresholds = np.arange(0.0, 1.05, 0.05)
    create_lineplot_averages_for_all_metrics_and_thresholds(file_name,
                                                            path_to_diagrams,
                                                            acc_df=metrics[0],
                                                            f1_df=metrics[1],
                                                            prec_df=metrics[2],
                                                            sens_df=metrics[3],
                                                            spec_df=metrics[4],
                                                            thresholds=thresholds)

    create_min_max_auc_roc_curve(file_name, path_to_predictions, path_to_diagrams, datasets)

    len_of_test_dataset = len(datasets[0])
    create_average_auc_roc_curve(file_name, path_to_predictions, path_to_diagrams, datasets, len_of_test_dataset)


def create_plots_for_column_excluded_datasets(file_name: str,
                                              path_to_predictions: str,
                                              path_to_metrics: str,
                                              path_to_diagrams: str) -> None:
    all_pred_dirs = os.listdir(path_to_predictions)
    all_metrics_dirs = os.listdir(path_to_metrics)

    all_pred_datasets = read_all_folders_files_in_memory(all_pred_dirs, path_to_predictions, is_prediction=True)

    # indexes --> 0: accuracy, 1: f1_score, 2: precision, 3: sensitivity, 4: specificity
    all_metrics_datasets = read_all_folders_files_in_memory(all_metrics_dirs, path_to_metrics, is_prediction=False)

    for col_excl_idx, pred_df_list in enumerate(all_pred_datasets):
        create_average_auc_roc_curve(file_name, path_to_predictions, path_to_diagrams, pred_df_list, len(pred_df_list[0]), col_excl_idx)

    thresholds = np.arange(0.0, 1.05, 0.05)

    for col_excl_idx, metric_df_list in enumerate(all_metrics_datasets):
        create_lineplot_averages_for_all_metrics_and_thresholds(file_name, path_to_diagrams, acc_df=metric_df_list[0],
                                                                f1_df=metric_df_list[1], prec_df=metric_df_list[2],
                                                                sens_df=metric_df_list[3], spec_df=metric_df_list[4],
                                                                thresholds=thresholds, col_excl_idx=col_excl_idx)
        # create_3d_plot_for_each_metrics(acc_df=metric_df_list[0], thresholds=thresholds, col_excl_idx=col_excl_idx)

    create_boxplot_for_col_excluded_datasets(datasets=all_pred_datasets, path_to_diagrams=path_to_diagrams)
    create_lineplot_for_one_metric_col_excluded_datasets(all_metrics_datasets, thresholds, path_to_diagrams)


if __name__ == '__main__':
    path_to_metrics = "C:/Egyetem_es_munka/Egyetem/MSc/Thesis/DataVisualisations/50rounds_default_randomF_10_bern05prob_0to1_thresh_20200925/metrics/"
    path_to_metrics_col_excluding = 'C:/Egyetem_es_munka/Egyetem/MSc/Thesis/DataVisualisations/50rounds_default_randomF_10_bern05prob_0to1_thresh_20200925/metrics/column_excluding/'
    path_to_predictions = "C:/Egyetem_es_munka/Egyetem/MSc/Thesis/DataVisualisations/50rounds_default_randomF_10_bern05prob_0to1_thresh_20200925/predictions/"
    path_to_predictions_col_excluding = "C:/Egyetem_es_munka/Egyetem/MSc/Thesis/DataVisualisations/50rounds_default_randomF_10_bern05prob_0to1_thresh_20200925/predictions/column_excluding/"
    path_to_diagrams = "C:/Egyetem_es_munka/Egyetem/MSc/Thesis/DataVisualisations/50rounds_default_randomF_10_bern05prob_0to1_thresh_20200925/diagrams/without_column_excluding/"
    path_to_diagrams_col_excl = "C:/Egyetem_es_munka/Egyetem/MSc/Thesis/DataVisualisations/50rounds_default_randomF_10_bern05prob_0to1_thresh_20200925/diagrams/column_excluding/separate_diagrams/"

    file_name = '50_simrounds_10_bern05prob'

    create_plots_for_without_column_excluded_datasets(file_name=file_name,
                                                      path_to_predictions=path_to_predictions,
                                                      path_to_metrics=path_to_metrics,
                                                      path_to_diagrams=path_to_diagrams)

    create_plots_for_column_excluded_datasets(file_name=file_name,
                                              path_to_predictions=path_to_predictions_col_excluding,
                                              path_to_metrics=path_to_metrics_col_excluding,
                                              path_to_diagrams=path_to_diagrams_col_excl)

