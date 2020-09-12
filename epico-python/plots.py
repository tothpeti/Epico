import os

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def create_boxplot_for_all_metrics_and_thresholds(name, path_to_diagrams,
                                                  acc_df, prec_df, f1_df,
                                                  spec_df, sens_df):
    cols_num = len(sens_df.columns)
    thresholds = [thresh[6:] + ' threshold' for thresh in sens_df.columns]
    # thresholds = [ thresh.split('_')[0] + ' threshold' for thresh in sens_df.columns ]
    for i in range(0, cols_num):
        tmp_data = {
            'accuracy': acc_df.iloc[:, i],
            'sensitivity': sens_df.iloc[:, i],
            'specificity': spec_df.iloc[:, i],
            'precision': prec_df.iloc[:, i],
            'f1_score': f1_df.iloc[:, i]
        }
        tmp_df = pd.DataFrame(data=tmp_data)

        plt.figure(figsize=(16, 6))
        plt.suptitle(thresholds[i], fontsize=18)
        plt.subplot(1, 2, 1)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        ax = sns.boxplot(data=tmp_df[['sensitivity', 'specificity', 'precision', 'f1_score']], orient="h",
                         palette="Set2")

        plt.subplot(1, 2, 2)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        ax = sns.boxplot(data=tmp_df[['accuracy']], orient="h", palette="Set2")

        #plt.show()

        file_name = name +'_boxplot_'+ thresholds[i]+'_all_metrics_all_covariates.png'
        plt.savefig(os.path.join(path_to_diagrams, file_name), bbox_inches='tight')


def create_lineplot_averages_for_all_metrics_and_thresholds(name, path_to_diagrams,
                                                            acc_df, prec_df,
                                                            f1_df, spec_df,
                                                            sens_df, thresholds):

    avg_acc_list = []
    avg_sens_list = []
    avg_spec_list = []
    avg_prec_list = []
    avg_f1_list = []

    for i in range(0, len(sens_df.columns)):
        avg_acc_list.append(acc_df.iloc[:, i].mean())
        avg_sens_list.append(sens_df.iloc[:, i].mean())
        avg_spec_list.append(spec_df.iloc[:, i].mean())
        avg_prec_list.append(prec_df.iloc[:, i].mean())
        avg_f1_list.append(f1_df.iloc[:, i].mean())

    tmp_all_metrics_df = pd.DataFrame()
    tmp_all_metrics_df['thresholds'] = pd.to_numeric(pd.Series(thresholds), errors='coerce')
    tmp_all_metrics_df['acc_mean'] = pd.to_numeric(pd.Series(avg_acc_list), errors='coerce')
    tmp_all_metrics_df['sens_mean'] = pd.to_numeric(pd.Series(avg_sens_list), errors='coerce')
    tmp_all_metrics_df['spec_mean'] = pd.to_numeric(pd.Series(avg_spec_list), errors='coerce')
    tmp_all_metrics_df['prec_mean'] = pd.to_numeric(pd.Series(avg_prec_list), errors='coerce')
    tmp_all_metrics_df['f1_mean'] = pd.to_numeric(pd.Series(avg_f1_list), errors='coerce')

    y_range = np.arange(0.0, 1.1, 0.1)

    ax = plt.figure(figsize=(12, 6))
    plt.title('Logistic regression considering all predictive variables', fontsize=16)
    plt.xticks(thresholds, rotation=70, fontsize=12)
    plt.yticks(y_range, fontsize=12)
    ax = sns.lineplot(data=tmp_all_metrics_df, x='thresholds', y='sens_mean', label='Sensitivity', palette="Set2")
    ax = sns.lineplot(data=tmp_all_metrics_df, x='thresholds', y='spec_mean', label='Specificity', palette="Set2")
    ax = sns.lineplot(data=tmp_all_metrics_df, x='thresholds', y='prec_mean', label='Precision', palette="Set2")
    ax = sns.lineplot(data=tmp_all_metrics_df, x='thresholds', y='f1_mean', label='F1 score', palette="Set2")
    ax = sns.lineplot(data=tmp_all_metrics_df, x='thresholds', y='acc_mean', label='Accuracy', palette="Set2")
    ax.legend(loc='best')
    ax.set_ylabel('The average value of quality metrics', fontsize=16)
    ax.set_xlabel('Threshold', fontsize=16)
    # ax.set(xlim=(float(thresholds[0]), float(thresholds[-1])))
    plt.xlim([float(thresholds[0])-0.025, float(thresholds[-1])+0.05])
    plt.ylim([0.0, 1.05])

    file_name = name +'_lineplots_all_average_metrics_all_thresholds_all_covariates.png'
    plt.savefig(os.path.join(path_to_diagrams, file_name), bbox_inches='tight')
    #plt.show()


def create_average_auc_roc_curve(name, path_to_predictions, path_to_diagrams, datasets_name, len_of_test_datasets):
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
    mean_fpr = np.linspace(0, 1, len_of_test_datasets)

    tpr_list = []
    auc_list = []

    for dataset in datasets_name:
        df = pd.read_csv(os.path.join(path_to_predictions, dataset))
        fpr, tpr, _ = roc_curve(df['y'], df['y_pred_probs'])
        roc_auc = roc_auc_score(df['y'], df['y_pred_probs'])

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tpr_list.append(interp_tpr)
        auc_list.append(roc_auc)

    mean_tpr = np.mean(tpr_list, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(auc_list)

    plt.figure( figsize=(12, 6))
    plt.plot(mean_fpr, mean_tpr, label=r'Mean ROC (AUC = %0.4f $\pm$ %0.3f)' % (mean_auc, std_auc))
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.025, 1.05])
    plt.ylim([-0.025, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Logistic regression considering all predictive variables')
    plt.legend(loc="lower right")
    #plt.show()

    file_name = name +'_average_roc_curve_all_covariates.png'
    plt.savefig(os.path.join(path_to_diagrams, file_name), bbox_inches='tight')


def create_min_max_auc_roc_curve(name, path_to_predictions, path_to_diagrams, datasets_name):

    results_list = []
    auc_list = []
    for i, dataset in enumerate(datasets_name):
        df = pd.read_csv(os.path.join(path_to_predictions, dataset))
        fpr, tpr, threshold = roc_curve(df['y'], df['y_pred_probs'])
        roc_auc = roc_auc_score(df['y'], df['y_pred_probs'])
        auc_list.append(roc_auc)
        results_list.append( [fpr, tpr, roc_auc, i] )

    max_auc_idx = auc_list.index(max(auc_list))
    min_auc_idx = auc_list.index(min(auc_list))

    print('min auc idx: '+str(results_list[min_auc_idx][3]+1))
    print('max auc idx: '+str(results_list[max_auc_idx][3]+1))

    plt.figure(figsize=(16, 6))
    ax = plt.subplot(1, 2, 1)
    plt.title('ROC curve - maximum AUC', fontsize=16)
    ax.plot(results_list[max_auc_idx][0], results_list[max_auc_idx][1], 'b', label='AUC = %0.4f' % results_list[max_auc_idx][2])
    ax.legend(loc='lower right', fontsize=12)
    ax.plot([0, 1], [0, 1], 'r--')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim([-0.025, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    # plt.show()

    ax2 = plt.subplot(1, 2, 2)
    plt.title('ROC curve - minimum AUC', fontsize=16)
    ax2.plot(results_list[min_auc_idx][0], results_list[min_auc_idx][1], 'b', label='AUC = %0.4f' % results_list[min_auc_idx][2])
    ax2.legend(loc='lower right', fontsize=12)
    ax2.plot([0, 1], [0, 1], 'r--')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim([-0.025, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    #plt.show()

    file_name = name +'_min_max_auc_roc_curve_all_covariates.png'
    plt.savefig(os.path.join(path_to_diagrams, file_name), bbox_inches='tight')
