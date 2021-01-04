"""
    Custom files
"""
from plots import create_plots_for_without_column_excluded_datasets, create_plots_for_column_excluded_datasets

if __name__ == '__main__':

    path_to_metrics = 'D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/DataVisualisations/Scenario5/logreg_0to1_thresholds/metrics/'
    path_to_metrics_col_excluding = 'D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/DataVisualisations/Scenario5/logreg_0to1_thresholds/metrics/column_excluding/'
    path_to_predictions = 'D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/DataVisualisations/Scenario5/logreg_0to1_thresholds/predictions/'
    path_to_predictions_col_excluding = 'D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/DataVisualisations/Scenario5/logreg_0to1_thresholds/predictions/column_excluding/'
    path_to_diagrams = 'D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/DataVisualisations/Scenario5/logreg_0to1_thresholds/diagrams/without_column_excluding/'
    path_to_diagrams_col_excluding = 'D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/DataVisualisations/Scenario5/logreg_0to1_thresholds/diagrams/column_excluding/'

    # If hyperparameter tuning ---> Tuned for roc_auc Random Forest
    title_name = 'RandomizedSearchCV tuned (for roc_auc) Random Forest'

    create_plots_for_without_column_excluded_datasets(title_name=title_name,
                                                      path_to_predictions=path_to_predictions,
                                                      path_to_metrics=path_to_metrics,
                                                      path_to_diagrams=path_to_diagrams)
    create_plots_for_column_excluded_datasets(title_name=title_name,
                                              path_to_predictions=path_to_predictions_col_excluding,
                                              path_to_metrics=path_to_metrics_col_excluding,
                                              path_to_diagrams=path_to_diagrams_col_excluding)

    print('Finished!')
