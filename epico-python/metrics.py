import os
from sklearn.metrics import confusion_matrix
import pandas as pd


def create_metrics(df, y_test, threshold_cols):
    accuracy_list = []
    f1_score_list = []
    precision_list = []
    sensitvity_list = []
    specificity_list = []

    for threshold_col in threshold_cols:
        conf_matrix = confusion_matrix(y_test, df[threshold_col])

        tn = conf_matrix[0][0]
        fp = conf_matrix[0][1]
        fn = conf_matrix[1][0]
        tp = conf_matrix[1][1]

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1_score = (2 * tp) / (2*tp + fp + fn)

        precision = tp / (tp + fp) if (tp+fp) != 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fn) != 0 else 0.0

        accuracy_list.append(round(accuracy, 3))
        f1_score_list.append(round(f1_score, 3))
        precision_list.append(round(precision, 3))
        sensitvity_list.append(round(sensitivity, 3))
        specificity_list.append(round(specificity, 3))

    return accuracy_list, f1_score_list, precision_list, sensitvity_list, specificity_list


def save_metrics(all_accuracy_list: list,
                 all_f1_score_list: list,
                 all_precision_list: list,
                 all_sensitivity_list: list,
                 all_specificity_list: list,
                 threshold_col_names: list,
                 path_to_save_folder: str,
                 excluded_col: str = "") -> None:

    # Convert lists to DataFrames
    accuracy_df = pd.DataFrame(all_accuracy_list, columns=threshold_col_names)
    f1_score_df = pd.DataFrame(all_f1_score_list, columns=threshold_col_names)
    precision_df = pd.DataFrame(all_precision_list, columns=threshold_col_names)
    sensitivity_df = pd.DataFrame(all_sensitivity_list, columns=threshold_col_names)
    specificity_df = pd.DataFrame(all_specificity_list, columns=threshold_col_names)

    # Save DataFrames into csv files
    accuracy_file_name = 'accuracy_'+excluded_col+'.csv'
    f1score_file_name = 'f1_score_'+excluded_col+'.csv'
    precision_file_name = 'precision_'+excluded_col+'.csv'
    sensitivity_file_name = 'sensitivity_'+excluded_col+'.csv'
    specificity_file_name = 'specificity_'+excluded_col+'.csv'

    accuracy_df.to_csv(os.path.join(path_to_save_folder, accuracy_file_name), sep=',', index=False)
    f1_score_df.to_csv(os.path.join(path_to_save_folder, f1score_file_name), sep=',', index=False)
    precision_df.to_csv(os.path.join(path_to_save_folder, precision_file_name), sep=',', index=False)
    sensitivity_df.to_csv(os.path.join(path_to_save_folder, sensitivity_file_name), sep=',', index=False)
    specificity_df.to_csv(os.path.join(path_to_save_folder, specificity_file_name), sep=',', index=False)


def save_prediction_df(df: pd.DataFrame,
                       dataset_name: str,
                       path: str) -> None:
    df.to_csv(os.path.join(path, dataset_name), sep=',', index=False)


def save_best_model_parameters(best_params_dict: dict,
                               dataset_name: str,
                               path: str) -> None:

    tmp_dict = {key:[value] for (key, value) in best_params_dict.items()}
    df = pd.DataFrame.from_dict(tmp_dict)
    df.to_csv(os.path.join(path, dataset_name), sep=',', index=False)
