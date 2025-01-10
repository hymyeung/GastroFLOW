import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.options.mode.chained_assignment = None


# function to deal with complex file names, please write your own regex to get case names
def find_case(n, match_pattern='(\d+)AH(\d+)'):
    # match_pattern = '(\d+)AH(\d+)'
    if match_pattern is None:
        return n
    else:
        match_groups = re.search(match_pattern, n)
        match_objs = [match_groups.group(1), match_groups.group(2)]
        corrected_match_obj = keep_digits(match_objs[1])
        case_num = match_objs[0] + 'AH' + corrected_match_obj
        return case_num


# function to help get case information from each id
def keep_digits(num_str, keep_num=5):
    str_len = len(num_str)
    if str_len < keep_num:
        num_str = '0' * (keep_num - str_len) + num_str
    elif str_len > keep_num:
        while str_len > keep_num:
            letter = num_str[0]
            if letter == '0':
                num_str = num_str[1:]
                str_len = len(num_str)
            else:
                break
    return num_str


# metric score func, the reason for score and prediction is that different thresholds are used
def metric_func(score, prediction, true_label):
    lbs = [0, 1]
    conf_mat = confusion_matrix(true_label, prediction, labels=lbs)
    tn = conf_mat[0][0]
    tp = conf_mat[1][1]
    fn = conf_mat[1][0]
    fp = conf_mat[0][1]
    p_num = tp + fn
    n_num = tn + fp

    epsilon = 1e-6
    tpr = float(tp + epsilon) / (tp + fn + epsilon)
    tnr = float(tn + epsilon) / (tn + fp + epsilon)
    ppv = float(tp + epsilon) / (tp + fp + epsilon)
    npv = float(tn + epsilon) / (tn + fn + epsilon)
    acc = float(tn + tp + epsilon) / (tn + tp + fp + fn + epsilon)
    f1 = f1_score(true_label, prediction)

    mcc = matthews_corrcoef(true_label, prediction)

    if len(list(set(true_label))) == 1:
        roc_auc = np.nan
    else:
        roc_auc = roc_auc_score(true_label, score)

    comb_metric = abs(1 - mcc) + abs(1 - npv) + abs(1 - tpr)

    metric_dict = {'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn, 'TPR': tpr,
                   'TNR': tnr, 'PPV': ppv, 'NPV': npv, 'ACC': acc, 'F1': f1, 'MCC': mcc, 'ROC_AUC': roc_auc,
                   'Manh Dist': comb_metric}
    return metric_dict


# help function to calculate the time save ratio of skipped cases
def calc_time_save(pd_predict_label, triage_plot_path=None):
    short_case = pd_predict_label.loc[:, 'case'].map(lambda x: x.split('Stomach ')[-1]).tolist()
    pd_predict_label['short case'] = short_case
    if 'case_lb' in pd_predict_label.columns:
        ca_idx = pd_predict_label['case_lb'] == 'CA'
        pd_ca_sec = pd_predict_label[ca_idx]
        pd_nca_sec = pd_predict_label[~ca_idx]
        pd_ca_sec = pd_ca_sec.sort_values(by='mean_score', ascending=False)
        pd_nca_sec = pd_nca_sec.sort_values(by='mean_score', ascending=False)
        pd_predict_label = pd.concat([pd_ca_sec, pd_nca_sec], axis=0).reset_index(drop=True)
    else:
        pd_predict_label = pd_predict_label.sort_values(by='mean_score', ascending=False).reset_index(drop=True)
    if triage_plot_path is not None:
        print()
    last_tp_index = pd_predict_label[pd_predict_label['label'] == 1].index
    last_pd_index = pd_predict_label[pd_predict_label['predict'] == 1].index
    first_neg_index = pd_predict_label[pd_predict_label['label'] == 0].index[0]
    if len(last_tp_index) > 0:
        last_tp_idx = last_tp_index[-1]
    else:
        last_tp_idx = 0
    if len(last_pd_index) > 0:
        last_pd_idx = last_pd_index[-1]
    else:
        last_pd_idx = 0

    # else:
    pd_len = len(pd_predict_label)
    pd_neg_len = len(pd_predict_label[pd_predict_label['label'] == 0])
    time_save_rate = min((pd_len - (last_pd_idx + 1)) / pd_neg_len, 1.0) * 100
    time_save_rate_theo = min(1.0, (pd_len - (last_tp_idx + 1)) / pd_len) * 100

    if last_pd_idx < last_tp_idx:
        time_save_rate = time_save_rate_theo

        # for i in reversed(last_tp_index):
        #     if last_pd_idx < i:
        #         print(f"{pd_predict_label.loc[i, ['case', 'logit']]}")

    print(
        f"last case: {pd_predict_label.loc[last_tp_idx, 'case']}, last score: {pd_predict_label.loc[last_tp_idx, 'mean_score']}")
    print(time_save_rate_theo)
    print(
        f"last case: {pd_predict_label.loc[last_pd_idx, 'case']}, last score: {pd_predict_label.loc[last_pd_idx, 'mean_score']}")
    print(time_save_rate)

    if triage_plot_path is not None:
        # label_dict = {1: "CA", 0: "NON-CA"}
        pd_predict_label['color'] = 'susp CA'
        pd_predict_label.loc[pd_predict_label['case_lb'] == 'CA', 'color'] = 'CA'
        if last_pd_idx + 1 < pd_len:
            pd_predict_label.iloc[last_pd_idx + 1:, -1] = 'benign'
        map_group_color = {"benign": 'cyan', "susp CA": 'orange', "CA": 'red'}
        pd_plot = pd_predict_label.iloc[::-1, :]
        ax = pd_plot.plot.barh(x="short case", y="mean_score", color=pd_plot.color.replace(map_group_color))
        ax.locator_params(axis='x', nbins=len(pd_predict_label))
        plt.xticks(np.arange(0, 1, step=.1))
        plt.yticks(fontsize=4)
        color_dict = {1: 'r', 0: 'k'}
        colors = pd_predict_label['label'].map(lambda x: color_dict[int(x)])
        for ytick, color in zip(ax.get_yticklabels(), colors[::-1]):
            ytick.set_color(color)
        legend_handles = [mpatches.Patch(color=color, label=group) for group, color in map_group_color.items()]
        ax.legend(handles=legend_handles)
        plt.title(triage_plot_path.split('/')[-1][:-4] + ' : ' + str(round(time_save_rate, 2)) + '%')
        plt.tight_layout()
        plt.savefig(triage_plot_path, dpi=300)
        plt.close()
    return {'TSR_P': time_save_rate, 'TSR_T': time_save_rate_theo}, pd_predict_label


