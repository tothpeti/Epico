import glob
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_all_datasets(path):
    os.chdir(path)
    return [file for file in glob.glob("*.csv")]


def create_metrics(df, y_test, threshold_cols):
    accuracy_list = []
    f1_score_list = []
    precision_list = []
    sensitvity_list = []
    specificity_list = []

    for threshold_col in threshold_cols:
        conf_matrix = confusion_matrix(y_test, df[threshold_col])

        tp = conf_matrix[0][0]
        fn = conf_matrix[0][1]
        fp = conf_matrix[1][0]
        tn = conf_matrix[1][1]

        #print(str(tp)+ " "+ str(fn)+ " "+ str(fp)+ " "+ str(tn))
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1_score = (2 * tp) / (2*tp + fp + fn)
        precision = tp / (tp + fp)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        accuracy_list.append(round(accuracy, 3))
        f1_score_list.append(round(f1_score, 3))
        precision_list.append(round(precision, 3))
        sensitvity_list.append(round(sensitivity, 3))
        specificity_list.append(round(specificity, 3))

    return accuracy_list, f1_score_list, precision_list, sensitvity_list, specificity_list


def save_metrics(all_accuracy_list, all_f1_score_list,
                 all_precision_list, all_sensitvity_list,
                 all_specificity_list, threshold_col_names,
                 path_to_save_folder):

    # Convert lists to DataFrames
    accuracy_df = pd.DataFrame(all_accuracy_list, columns=threshold_col_names)
    f1_score_df = pd.DataFrame(all_f1_score_list, columns=threshold_col_names)
    precision_df = pd.DataFrame(all_precision_list, columns=threshold_col_names)
    sensitivity_df = pd.DataFrame(all_sensitvity_list, columns=threshold_col_names)
    specificity_df = pd.DataFrame(all_specificity_list, columns=threshold_col_names)

    # Save DataFrames into csv files
    accuracy_df.to_csv(os.path.join(path_to_save_folder, r'accuracy.csv'), sep=',', index=False)
    f1_score_df.to_csv(os.path.join(path_to_save_folder, r'f1_score.csv'), sep=',', index=False)
    precision_df.to_csv(os.path.join(path_to_save_folder, r'precision.csv'), sep=',', index=False)
    sensitivity_df.to_csv(os.path.join(path_to_save_folder, r'sensitivity.csv'), sep=',', index=False)
    specificity_df.to_csv(os.path.join(path_to_save_folder, r'specificity.csv'), sep=',', index=False)


def plot_roc_curve(df, y_test):
    logit_roc_auc = roc_auc_score(y_test, df['y_pred0.5'])
    fpr, tpr, thresholds = roc_curve(y_test, df['y_pred0.5'])
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    # plt.savefig('Log_ROC')
    plt.show()

if __name__ == '__main__':
    path_to_files = "C:/Egyetem_es_munka/Egyetem/MSc/Thesis/DataVisualisations/50rounds_10_bern05prob_with_05_to_095_thresholds_TEST/datasets/"
    path_to_save_folder = "C:/Egyetem_es_munka/Egyetem/MSc/Thesis/DataVisualisations/50rounds_10_bern05prob_with_05_to_095_thresholds_TEST/"
    datasets = get_all_datasets(path_to_files)

    """
        Preprocess and model train/test
    """
    # Start(inclusive), End(exclusive), Steps
    thresholds = np.arange(0.5, 1, 0.05)
    # Need to round the numbers to 2 decimal places, due to decimal error when try to print out the value
    thresholds = [round(threshold, 2) for threshold in thresholds]
    threshold_col_names = ['y_pred'+str(threshold) for threshold in thresholds]

    all_accuracy_list = []
    all_f1_score_list = []
    all_precision_list = []
    all_sensitvity_list = []
    all_specificity_list = []

    result_df = pd.DataFrame()
    for dataset in datasets:
        path_to_dataset = path_to_files + dataset
        df = pd.read_csv(path_to_dataset)

        x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, 0:-1],
                                                            df['y'],
                                                            test_size=0.3,
                                                            random_state=0)

        logistic_reg = LogisticRegression()

        # Train mode
        logistic_reg.fit(x_train, y_train)

        # Test model
        #y_pred = logistic_reg.predict(x_test)
        result_df['y_pred_probs'] = logistic_reg.predict_proba(x_test)[:, 1]
        #df.loc[x_test.index, 'y_pred_probs'] = logistic_reg.predict_proba(x_test)[:, 1]

        for idx in range(len(thresholds)):
            result_df[threshold_col_names[idx]] = np.nan
            result_df.loc[result_df['y_pred_probs'] >= thresholds[idx], threshold_col_names[idx]] = 1
            result_df.loc[result_df['y_pred_probs'] < thresholds[idx], threshold_col_names[idx]] = 0

        accuracy_list, f1_score_list, precision_list, sensitivity_list, specificity_list = create_metrics(result_df, y_test, threshold_col_names)
        plot_roc_curve(result_df, y_test)

        all_accuracy_list.append(accuracy_list)
        all_f1_score_list.append(f1_score_list)
        all_precision_list.append(precision_list)
        all_sensitvity_list.append(sensitivity_list)
        all_specificity_list.append(specificity_list)
"""
    save_metrics(all_accuracy_list, all_f1_score_list,
                 all_precision_list, all_sensitvity_list,
                 all_specificity_list, threshold_col_names,
                 path_to_save_folder)
"""