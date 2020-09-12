import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

"""
    Custom files
"""
from metrics import create_metrics, save_metrics, save_prediction_df


def run_logistic_reg(features, target, thresholds, threshold_col_names, test_size=0.3):

    result_df = pd.DataFrame()

    x_train, x_test, y_train, y_test = train_test_split(features,
                                                        target,
                                                        test_size=test_size,
                                                        random_state=0)

    # Initialize and train model
    logistic_reg = LogisticRegression().fit(x_train, y_train)

    # Test model
    result_df = pd.concat([result_df, x_test, y_test], axis=1)
    result_df.reset_index(inplace=True, drop=True)

    result_df['y_pred_probs'] = logistic_reg.predict_proba(x_test)[:, 1]
    result_df['y_pred_probs'] = result_df['y_pred_probs'].round(decimals=3)

    # Create y_predN columns by using threshold values
    for idx in range(len(thresholds)):
        result_df[threshold_col_names[idx]] = np.nan
        result_df.loc[result_df['y_pred_probs'] >= thresholds[idx], threshold_col_names[idx]] = 1
        result_df.loc[result_df['y_pred_probs'] < thresholds[idx], threshold_col_names[idx]] = 0

    return result_df, y_test


def run_process_without_column_excluding(datasets, datasets_names, thresholds, threshold_col_names,
                                         path_to_predictions, path_to_metrics):
    all_accuracy_list = []
    all_f1_score_list = []
    all_precision_list = []
    all_sensitivity_list = []
    all_specificity_list = []

    for idx, df in enumerate(datasets):
        features = df.drop(columns=['y'], axis=1)
        target = df['y']
        result_df, y_test = run_logistic_reg(features, target, thresholds, threshold_col_names, test_size=0.3)

        accuracy_list, f1_score_list, precision_list, sensitivity_list, specificity_list = create_metrics(result_df,
                                                                                                          y_test,
                                                                                                          threshold_col_names)

        prediction_file_name = datasets_names[idx]
        save_prediction_df(result_df, prediction_file_name, path_to_predictions)

        all_accuracy_list.append(accuracy_list)
        all_f1_score_list.append(f1_score_list)
        all_precision_list.append(precision_list)
        all_sensitivity_list.append(sensitivity_list)
        all_specificity_list.append(specificity_list)

    save_metrics(all_accuracy_list, all_f1_score_list,
                 all_precision_list, all_sensitivity_list,
                 all_specificity_list, threshold_col_names,
                 path_to_metrics)


def run_process_with_column_excluding(num_of_cols, datasets, datasets_names, thresholds, threshold_col_names,
                                      path_to_predictions_col_excluding, path_to_metrics_col_excluding):

    all_accuracy_list = []
    all_f1_score_list = []
    all_precision_list = []
    all_sensitivity_list = []
    all_specificity_list = []

    for col_to_exclude in range(num_of_cols):
        for idx, df in enumerate(datasets):

            col_name_to_exclude = 'x'+str(col_to_exclude+1)
            features = df.drop(columns=[col_name_to_exclude, 'y'], axis=1)
            target = df['y']
            result_df, y_test = run_logistic_reg(features, target, thresholds, threshold_col_names, test_size=0.3)

            accuracy_list, f1_score_list, precision_list, sensitivity_list, specificity_list = create_metrics(result_df, y_test, threshold_col_names)

            prediction_file_name = datasets_names[idx].split('.')[0]+'_'+str(col_to_exclude)+'.csv'
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
