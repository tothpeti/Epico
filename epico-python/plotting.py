"""
    Custom files
"""
from plots import create_plots_for_without_column_excluded_datasets, create_plots_for_column_excluded_datasets

if __name__ == '__main__':
    # Home PC

    path_to_metrics = 'D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/DataVisualisations/Scenario5/logreg_0to1_thresholds/metrics/'
    path_to_metrics_col_excluding = 'D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/DataVisualisations/Scenario5/logreg_0to1_thresholds/metrics/column_excluding/'
    path_to_predictions = 'D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/DataVisualisations/Scenario5/logreg_0to1_thresholds/predictions/'
    path_to_predictions_col_excluding = 'D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/DataVisualisations/Scenario5/logreg_0to1_thresholds/predictions/column_excluding/'
    path_to_diagrams = 'D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/DataVisualisations/Scenario5/logreg_0to1_thresholds/diagrams/without_column_excluding/'
    path_to_diagrams_col_excluding = 'D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/DataVisualisations/Scenario5/logreg_0to1_thresholds/diagrams/column_excluding/'
    """
    # Laptop
    path_to_metrics = "C:/Egyetem_es_munka/Egyetem/MSc/Thesis/DataVisualisations/Scenario2/tuned_ROC_random_forest_0to1_threshold/metrics/"
    path_to_metrics_col_excluding = 'C:/Egyetem_es_munka/Egyetem/MSc/Thesis/DataVisualisations/Scenario2/tuned_ROC_random_forest_0to1_threshold/metrics/column_excluding/'
    path_to_predictions = "C:/Egyetem_es_munka/Egyetem/MSc/Thesis/DataVisualisations/Scenario2/tuned_ROC_random_forest_0to1_threshold/predictions/"
    path_to_predictions_col_excluding = "C:/Egyetem_es_munka/Egyetem/MSc/Thesis/DataVisualisations/Scenario2/tuned_ROC_random_forest_0to1_threshold/predictions/column_excluding/"
    path_to_diagrams = "C:/Egyetem_es_munka/Egyetem/MSc/Thesis/DataVisualisations/Scenario2/tuned_ROC_random_forest_0to1_threshold/diagrams/without_column_excluding/"
    path_to_diagrams_col_excluding = "C:/Egyetem_es_munka/Egyetem/MSc/Thesis/DataVisualisations/Scenario2/tuned_ROC_random_forest_0to1_threshold/diagrams/column_excluding/"
    """
    # Ha hyperparameter tuning ---> Tuned for roc_auc Random Forest
    title_name = 'RandomizedSearchCV tuned (for roc_auc) Random Forest'
    # title_name = 'Default Random Forest'
    #title_name = 'Logistic Regression'

    create_plots_for_without_column_excluded_datasets(title_name=title_name,
                                                      path_to_predictions=path_to_predictions,
                                                      path_to_metrics=path_to_metrics,
                                                      path_to_diagrams=path_to_diagrams)
    """
    create_plots_for_column_excluded_datasets(title_name=title_name,
                                              path_to_predictions=path_to_predictions_col_excluding,
                                              path_to_metrics=path_to_metrics_col_excluding,
                                              path_to_diagrams=path_to_diagrams_col_excluding)
    """
    print('Finished!')
