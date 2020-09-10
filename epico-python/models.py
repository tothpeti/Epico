import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


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