"""
    Custom files
"""
from plots import create_plots_for_without_column_excluded_datasets, create_plots_for_column_excluded_datasets

if __name__ == '__main__':
    # Home PC
    path_to_datasets = 'D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/Generated_Data_Visualizations/50rounds_default_randomF_10_bern05prob_0to1_thresh_20200928/datasets/'
    path_to_metrics = 'D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/Generated_Data_Visualizations/50rounds_default_randomF_10_bern05prob_0to1_thresh_20200928/metrics/'
    path_to_metrics_col_excluding = 'D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/Generated_Data_Visualizations/50rounds_default_randomF_10_bern05prob_0to1_thresh_20200928/metrics/column_excluding/'
    path_to_predictions = 'D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/Generated_Data_Visualizations/50rounds_default_randomF_10_bern05prob_0to1_thresh_20200928/predictions/'
    path_to_predictions_col_excluding = 'D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/Generated_Data_Visualizations/50rounds_default_randomF_10_bern05prob_0to1_thresh_20200928/predictions/column_excluding/'
    path_to_diagrams = 'D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/Generated_Data_Visualizations/50rounds_default_randomF_10_bern05prob_0to1_thresh_20200928/diagrams/'
    path_to_diagrams_col_excluding = 'D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/Generated_Data_Visualizations/50rounds_default_randomF_10_bern05prob_0to1_thresh_20200928/diagrams/column_excluding'

    # Laptop
    # path_to_metrics = "C:/Egyetem_es_munka/Egyetem/MSc/Thesis/DataVisualisations/50rounds_tuned_randomF_10_bern05prob_0to1_thresh_20200930/metrics/"
    # path_to_metrics_col_excluding = 'C:/Egyetem_es_munka/Egyetem/MSc/Thesis/DataVisualisations/50rounds_tuned_randomF_10_bern05prob_0to1_thresh_20200930/metrics/column_excluding/'
    # path_to_predictions = "C:/Egyetem_es_munka/Egyetem/MSc/Thesis/DataVisualisations/50rounds_tuned_randomF_10_bern05prob_0to1_thresh_20200930/predictions/"
    # path_to_predictions_col_excluding = "C:/Egyetem_es_munka/Egyetem/MSc/Thesis/DataVisualisations/50rounds_tuned_randomF_10_bern05prob_0to1_thresh_20200930/predictions/column_excluding/"
    # path_to_diagrams = "C:/Egyetem_es_munka/Egyetem/MSc/Thesis/DataVisualisations/50rounds_tuned_randomF_10_bern05prob_0to1_thresh_20200930/diagrams/without_column_excluding/"
    # path_to_diagrams_col_excl = "C:/Egyetem_es_munka/Egyetem/MSc/Thesis/DataVisualisations/50rounds_tuned_randomF_10_bern05prob_0to1_thresh_20200930/diagrams/column_excluding/"

    file_name = '50rounds_10_bern05prob'

    """
    create_plots_for_without_column_excluded_datasets(file_name=file_name,
                                                      path_to_predictions=path_to_predictions,
                                                      path_to_metrics=path_to_metrics,
                                                      path_to_diagrams=path_to_diagrams)
    """
    create_plots_for_column_excluded_datasets(file_name=file_name,
                                              path_to_predictions=path_to_predictions_col_excluding,
                                              path_to_metrics=path_to_metrics_col_excluding,
                                              path_to_diagrams=path_to_diagrams_col_excluding)