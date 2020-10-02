from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np

"""
    Custom files
"""
from metrics import create_metrics, save_metrics, save_prediction_df


def run_model(model,
              features,
              target,
              thresholds,
              threshold_col_names,
              test_size=0.3):

    # Split dataset
    x_train, x_test, y_train, y_test = train_test_split(features,
                                                        target,
                                                        test_size=test_size,
                                                        random_state=0)

    # Initialize and train model
    trained_model = model.fit(x_train, y_train)

    # Test model
    return test_model(trained_model, x_test, y_test, thresholds, threshold_col_names)


def test_model(trained_model,
               x_test,
               y_test,
               thresholds,
               threshold_col_names):

    # Combine x_test, and y_test into one dataframe
    result_df = pd.DataFrame()
    result_df = pd.concat([result_df, x_test, y_test], axis=1)
    result_df.reset_index(inplace=True, drop=True)

    result_df['y_pred_probs'] = trained_model.predict_proba(x_test)[:, 1]
    result_df['y_pred_probs'] = result_df['y_pred_probs'].round(decimals=3)

    # Create y_predN columns by using threshold values
    for idx in range(len(thresholds)):
        result_df[threshold_col_names[idx]] = np.nan
        result_df.loc[result_df['y_pred_probs'] >= thresholds[idx], threshold_col_names[idx]] = 1
        result_df.loc[result_df['y_pred_probs'] < thresholds[idx], threshold_col_names[idx]] = 0

    return result_df, y_test


def run_without_column_excluding(model,
                                 datasets: list,
                                 datasets_names: list,
                                 thresholds: list,
                                 threshold_col_names: list,
                                 path_to_predictions: str,
                                 path_to_metrics: str) -> None:

    all_accuracy_list = []
    all_f1_score_list = []
    all_precision_list = []
    all_sensitivity_list = []
    all_specificity_list = []

    for idx, df in enumerate(datasets):
        features = df.drop(columns=['y'], axis=1)
        target = df['y']
        result_df, y_test = run_model(model=model,
                                      features=features,
                                      target=target,
                                      thresholds=thresholds,
                                      threshold_col_names=threshold_col_names,
                                      test_size=0.3)

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


def run_with_column_excluding(model,
                              num_of_cols: int,
                              datasets: list,
                              datasets_names: list,
                              thresholds: list,
                              threshold_col_names: list,
                              path_to_predictions_col_excluding: str,
                              path_to_metrics_col_excluding: str) -> None:

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
            result_df, y_test = run_model(model=model,
                                          features=features,
                                          target=target,
                                          thresholds=thresholds,
                                          threshold_col_names=threshold_col_names, test_size=0.3)

            accuracy_list, f1_score_list, precision_list, sensitivity_list, specificity_list = create_metrics(result_df,
                                                                                                              y_test,
                                                                                                              threshold_col_names)

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


def run_with_hyperparameter_search_and_column_excluding(model,
                                                        datasets: list,
                                                        datasets_names: list,
                                                        thresholds: list,
                                                        threshold_col_names: list,
                                                        num_of_cols: int,
                                                        path_to_predictions: str,
                                                        path_to_metrics: str,
                                                        model_params: dict = None):

    all_accuracy_list = []
    all_f1_score_list = []
    all_precision_list = []
    all_sensitivity_list = []
    all_specificity_list = []

    # Predict on all the datasets separately by using the best model
    for col_to_exclude in range(num_of_cols):
        for idx, df in enumerate(datasets):

            col_name_to_exclude = 'x'+str(col_to_exclude+1)
            features = df.drop(columns=[col_name_to_exclude, 'y'], axis=1)
            target = df['y']

            x_train, x_test, y_train, y_test = train_test_split(features,
                                                                target,
                                                                test_size=0.3,
                                                                random_state=0)
            # Run hyperparameter search
            clf = RandomizedSearchCV(model, model_params, n_iter=50, cv=10, verbose=0, random_state=0, n_jobs=-1)
            best_model = clf.fit(x_train, y_train)
            print(best_model.best_params_)

            result_df, y_test = test_model(trained_model=best_model,
                                           x_test=x_test,
                                           y_test=y_test,
                                           thresholds=thresholds,
                                           threshold_col_names=threshold_col_names)

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

