import glob
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import pandas as pd
import numpy as np


def get_all_datasets(path):
    os.chdir(path)
    return [file for file in glob.glob("*.csv")]


if __name__ == '__main__':
    path_to_files = "C:/Egyetem_es_munka/Egyetem/MSc/Thesis/DataVisualisations/50rounds_10_bern05prob_with_05_to_095_thresholds_TEST/datasets/"
    path_to_save_folder = "C:/Egyetem_es_munka/Egyetem/MSc/Thesis/DataVisualisations/50rounds_10_bern05prob_with_05_to_095_thresholds_TEST/"
    datasets = get_all_datasets(path_to_files)

    """
        Preprocess and model train/test
    """
    # Start(inclusive), End(exclusive), Steps
    thresholds = np.arange(0.5, 1, 0.05)

    for dataset in datasets:
        path_to_dataset = path_to_files + dataset
        df = pd.read_csv(path_to_dataset)

        x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, 0:len(df.columns)-1],
                                                            df['y'],
                                                            test_size=0.3,
                                                            random_state=0)

        logistic_reg = LogisticRegression()

        # Train model
        logistic_reg.fit(x_train, y_train)

        # Test model
        # y_pred = logistic_reg.predict(x_test)
        df['y_pred_probs'] = logistic_reg.predict_proba(x_test)

        threshold_cols = []
        for threshold in thresholds:
            col_name = 'y_pred'+str(threshold)
            threshold_cols.append(col_name)

            df[col_name] = 0
            df.loc[df['y_pred_probs'] >= threshold, col_name] = 1
            df.loc[df['y_pred_probs'] < threshold, col_name] = 0

