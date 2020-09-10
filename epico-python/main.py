import glob
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from metrics import create_metrics, save_metrics, save_prediction_df


def get_all_datasets(path):
    os.chdir(path)
    return [file for file in glob.glob("*.csv")]


if __name__ == '__main__':
    path_to_datasets = "D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/Generated_Data_Visualizations/50rounds_10_bern05prob_with_0_to_1_thresholds_20200903/datasets/"
    path_to_save_folder = "D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/Generated_Data_Visualizations/50rounds_10_bern05prob_with_0_to_1_thresholds_20200903/"
    path_to_predictions = "D:/Egyetem/MSc/TDK_Diploma_dolgozat/MasterThesis/Generated_Data_Visualizations/50rounds_10_bern05prob_with_0_to_1_thresholds_20200903/predictions/"
    datasets = get_all_datasets(path_to_datasets)

    """
        Preprocess and model train/test
    """
    # Start(inclusive), End(exclusive), Steps
    # 0.0 - 1.0
    thresholds = np.arange(0.0, 1.05, 0.05)

    # Need to round the numbers to 2 decimal places, due to decimal error when try to print out the value
    thresholds = [round(threshold, 2) for threshold in thresholds]
    threshold_col_names = ['y_pred'+str(threshold) for threshold in thresholds]

    all_accuracy_list = []
    all_f1_score_list = []
    all_precision_list = []
    all_sensitivity_list = []
    all_specificity_list = []

    for dataset in datasets:
        result_df = pd.DataFrame()

        # path_to_dataset = path_to_files + dataset
        df = pd.read_csv(os.path.join(path_to_datasets, dataset))

        x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, 0:-1],
                                                            df['y'],
                                                            test_size=0.3,
                                                            random_state=0)

        logistic_reg = LogisticRegression()

        # Train mode
        logistic_reg.fit(x_train, y_train)

        # Test model
        #y_pred = logistic_reg.predict(x_test)
        result_df = pd.concat([result_df, x_test, y_test], axis=1)
        result_df.reset_index(inplace=True, drop=True)

        # result_df['y'] = y_test
        result_df['y_pred_probs'] = logistic_reg.predict_proba(x_test)[:, 1]
        result_df['y_pred_probs'] = result_df['y_pred_probs'].round(decimals=3)
        #df.loc[x_test.index, 'y_pred_probs'] = logistic_reg.predict_proba(x_test)[:, 1]

        # Create y_predN columns by using threshold values
        for idx in range(len(thresholds)):
            result_df[threshold_col_names[idx]] = np.nan
            result_df.loc[result_df['y_pred_probs'] >= thresholds[idx], threshold_col_names[idx]] = 1
            result_df.loc[result_df['y_pred_probs'] < thresholds[idx], threshold_col_names[idx]] = 0

        accuracy_list, f1_score_list, precision_list, sensitivity_list, specificity_list = create_metrics(result_df, y_test, threshold_col_names)

        save_prediction_df(result_df, dataset, path_to_predictions)

        all_accuracy_list.append(accuracy_list)
        all_f1_score_list.append(f1_score_list)
        all_precision_list.append(precision_list)
        all_sensitivity_list.append(sensitivity_list)
        all_specificity_list.append(specificity_list)

    save_metrics(all_accuracy_list, all_f1_score_list,
                 all_precision_list, all_sensitivity_list,
                 all_specificity_list, threshold_col_names,
                 path_to_save_folder)
