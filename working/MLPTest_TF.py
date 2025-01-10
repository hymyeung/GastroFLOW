#!/usr/bin/env python
# coding: utf-8

import copy
import math
import os
from scipy import stats
import shutil
import time
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from Tool import metric_func, calc_time_save


class Config:
    dataset_lst = [
                    'train_data',
                    'sample_data',
                   ]
    key_features = [
                    'Nucleus: Area', 'Nucleus: Perimeter', 'Nucleus: Circularity', 'Nucleus: Max caliper',
                    'Nucleus: Min caliper', 'Nucleus: Eccentricity', 'Nucleus: Hematoxylin OD mean',
                    'Nucleus: Hematoxylin OD sum',
                    'Nucleus: Hematoxylin OD std dev', 'Nucleus: Hematoxylin OD max', 'Nucleus: Hematoxylin OD min',
                    'Nucleus: Hematoxylin OD range',
                    'Nucleus: Eosin OD mean', 'Nucleus: Eosin OD sum', 'Nucleus: Eosin OD std dev',
                    'Nucleus: Eosin OD max',
                    'Nucleus: Eosin OD min', 'Nucleus: Eosin OD range', 'Cell: Area', 'Cell: Perimeter',
                    'Cell: Circularity', 'Cell: Max caliper', 'Cell: Min caliper',
                    'Cell: Eccentricity', 'Cell: Hematoxylin OD mean', 'Cell: Hematoxylin OD std dev',
                    'Cell: Hematoxylin OD max', 'Cell: Hematoxylin OD min', 'Cell: Eosin OD mean',
                    'Cell: Eosin OD std dev',
                    'Cell: Eosin OD max', 'Cell: Eosin OD min', 'Cytoplasm: Hematoxylin OD mean',
                    'Cytoplasm: Hematoxylin OD std dev', 'Cytoplasm: Hematoxylin OD max',
                    'Cytoplasm: Hematoxylin OD min',
                    'Cytoplasm: Eosin OD mean', 'Cytoplasm: Eosin OD std dev', 'Cytoplasm: Eosin OD max',
                    'Cytoplasm: Eosin OD min', 'Nucleus/Cell area ratio'
                    ]
    label_column = 'Import_diagnosis'
    lb_dict = {'N': 0, 'C': 1}
    tgt_thresholds = "0.6_0.5_0.2"
    load_scaler = True
    SAVE_TRIAGE = True
    GEN_THERMAL = False
    SAVE_ROC_CURVE_PLOT = True
    PATCH_ONLY = False


if __name__ == "__main__":
    # setting random seed
    random_seed = 42
    np.random.seed(random_seed)

    # loading dataset
    train_dataset = Config.dataset_lst[0]
    test_dataset_lst = Config.dataset_lst[1]
    if not isinstance(test_dataset_lst, list):
        test_dataset_lst = [test_dataset_lst]

    exclude_cases = []

    # preparing models
    model_sav_dir = f"../ckpt/{train_dataset}/"
    all_model_files = glob(model_sav_dir + '*.h5')
    assert len(all_model_files) >= 1

    sc_model_file = os.path.join(model_sav_dir, 'scaler.sav')
    if os.path.exists(sc_model_file):
        sc = joblib.load(sc_model_file)
    else:
        Config.load_scaler = False

    # infer on test datasets
    for test_dataset in test_dataset_lst:
        print(test_dataset)

        # set up input dataset and output dirs
        cellular_feature_dir = f"../dataset/feature_dataset/{test_dataset}/"
        output_dir = f"../result/{test_dataset}/tensorflow/"

        # pd_train_data = pd.read_csv(f"../dataset/feature_dataset/{train_dataset}/aggregated_data.csv")
        # pd_train_data = pd_train_data[(np.abs(stats.zscore(pd_train_data.loc[:, Config.key_features])) < 3).all(axis=1)]

        pd_patch_score_rec = pd.DataFrame([])
        pd_case_score_rec = pd.DataFrame([])
        
        output_dir = os.path.join(output_dir, 'infer/')
        shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir, exist_ok=True)

        # score prediction
        case_time = 0
        patch_time = 0
        pd_Y_case_label = None
        model_sav_lst = []
        for PATCH_DATA in [False, True]:
            if PATCH_DATA:
                external_cellular_feature_path = os.path.join(cellular_feature_dir,
                                                              'aggregated_patch_data_500.csv')
                method_keyword = 'by_patches'
                if not os.path.exists(external_cellular_feature_path):
                    pd_patch_score_rec = pd_case_score_rec
                    break
            else:
                external_cellular_feature_path = os.path.join(cellular_feature_dir, 'aggregated_data.csv')
                method_keyword = 'by_image'

            pd_data = pd.read_csv(external_cellular_feature_path).drop_duplicates()
            all_cases = np.array(pd_data['case'])
            data_ids = pd_data['id'].values
            X_dataset = pd_data.loc[:, Config.key_features].values
            Y_dataset = pd_data.loc[:, Config.label_column].map(Config.lb_dict).values
            np_tmp = np.concatenate([np.expand_dims(all_cases, axis=-1), np.expand_dims(Y_dataset, axis=-1)], axis=-1)
            pd_Y_label = pd.DataFrame(np_tmp, columns=['case', 'label'])
            pd_Y_label = pd_Y_label[~pd_Y_label['case'].isin(exclude_cases)]
            pd_Y_case_label = pd_Y_label.groupby('case').agg(pd.Series.mode).rename_axis('case').reset_index()
            Y_case_label = pd_Y_case_label['label']

            time_start = time.time()

            # scaler normalization
            if not Config.load_scaler:
                sc = StandardScaler()
                X_dataset = sc.fit_transform(X_dataset)
            else:
                X_dataset = sc.transform(X_dataset)

            # inferring prediction
            model_pred_lst = []
            model_sav_lst = [file for file in all_model_files if file.endswith('.h5')]
            score_dict = dict()
            score_dict['case'] = all_cases

            for i, model_sav in enumerate(model_sav_lst):
                model = keras.models.load_model(model_sav)
                Y_pred = model.predict(X_dataset)
                Y_pred = np.array(Y_pred).squeeze().astype(float)
                score_dict[f'model_{i}'] = Y_pred

            time_end = time.time()

            if PATCH_DATA:
                patch_time = time_end - time_start
                print(f"time for patch prediction is {patch_time}")
            else:
                case_time = time_end - time_start
                print(f"time for case prediction is {case_time}")

            pd_score = pd.DataFrame(score_dict)
            pd_score.to_csv(output_dir + f"score_{method_keyword}.csv", index=False)

            if PATCH_DATA:
                pd_patch_score_rec = pd_score
            else:
                pd_case_score_rec = pd_score

        # post-processing and result generation
        case_predict_path = os.path.join(output_dir, 'case_pred.csv')
        patch_predict_path = os.path.join(output_dir, 'patch_pred.csv')
        pd_case_score_rec = pd_case_score_rec[~pd_case_score_rec['case'].isin(exclude_cases)]
        pd_patch_score_rec = pd_patch_score_rec[~pd_patch_score_rec['case'].isin(exclude_cases)]
        pd_case_score_rec.to_csv(case_predict_path, index=False)
        pd_patch_score_rec.to_csv(patch_predict_path, index=False)

        # calculate scores under different thresholds
        th1_lst = list(np.linspace(0.1, 1.0, 9, endpoint=False))
        th2_lst = [0.5]
        th3_lst = list(np.linspace(0.1, 1.0, 9, endpoint=False))
        th1_lst = [round(elem, 2) for elem in th1_lst]
        th2_lst = [round(elem, 2) for elem in th2_lst]
        th3_lst = [round(elem, 2) for elem in th3_lst]
        comb_th_lst = [[x1, x2, x3] for x1 in th1_lst for x2 in th2_lst for x3 in th3_lst]

        record_dict_lst = list()
        for th_comb in tqdm(comb_th_lst, total=len(comb_th_lst)):
            print(th_comb)
            Y_case_label = pd_Y_case_label['label']
            pd_case_score = copy.deepcopy(pd_case_score_rec)
            pd_patch_score = copy.deepcopy(pd_patch_score_rec)

            record_dict = dict()
            md_score_th, pos_rate_th, pos_num_th = th_comb
            record_dict['md_score_th'] = md_score_th
            record_dict['pos_rate_th'] = pos_rate_th
            record_dict['pos_num_th'] = pos_num_th
            th_comb_name = '_'.join([str(x) for x in th_comb])

            model_num = len(model_sav_lst)

            # md_score_th: threshold for model prediction
            pd_case_score = pd_case_score.groupby('case').agg(max).rename_axis('case').reset_index()
            pd_case_score['pos_cnt'] = (pd_case_score.loc[:, pd_case_score.columns.str.contains('model')].
                                        applymap(lambda x: 1 if round(x, 2) >= md_score_th else 0).apply(sum, axis=1))
            pd_case_score['mean_score'] = pd_case_score.loc[:, pd_case_score.columns.str.contains('model')].apply(
                np.mean,
                axis=1)

            # pos_rate_th: threshold for model num ratio of positive prediction
            threshold_num = math.ceil(pos_rate_th * model_num)
            pd_case_score['predict'] = pd_case_score['pos_cnt'].map(lambda x: 1 if x >= threshold_num else 0)
            print(f"cases positive: {len(pd_case_score[pd_case_score['predict'] == 1])}/{len(pd_case_score)}")

            sr_case_num = np.array(pd_patch_score.groupby('case').size())
            model_columns = pd_patch_score.columns.str.contains('model')
            pd_md_num = len(pd_patch_score.columns.str.contains('model'))
            pd_patch_score['pos_md_cnt'] = (pd_patch_score.loc[:, model_columns].
                                            applymap(lambda x: 1 if x >= md_score_th else 0).apply(sum, axis=1))
            pd_patch_score['patch_predict'] = pd_patch_score['pos_md_cnt'].apply(
                lambda x: 1 if x >= threshold_num else 0)
            pos_case_cnt = np.array(pd_patch_score.groupby('case')['patch_predict'].apply(sum))

            # pos_num_th: threshold for ratio of positive patches
            model_columns = pd_patch_score.columns.str.contains('model')
            mean_score_lst = list()
            case_lst = []
            for case in pd_patch_score['case'].drop_duplicates():
                case_lst.append(case)
                pd_patch_score_sec = pd_patch_score[pd_patch_score['case'] == case]
                model_mean_value = np.array(pd_patch_score_sec.loc[:, model_columns].mean(axis=1))
                case_cnt = np.floor(len(pd_patch_score_sec) * pos_num_th).astype(int)
                mean_score = np.mean(np.sort(model_mean_value)[-1 * case_cnt:])
                mean_score_lst.append(mean_score)
            pd_patch_mean_score = pd.DataFrame({'case': case_lst, 'mean_score': mean_score_lst})
            pd_patch_score = pd_patch_score.groupby('case').aggregate(np.mean).rename_axis('case').reset_index()
            pd_patch_score['case_cnt'] = sr_case_num
            pd_patch_score['pos_patch_cnt'] = pos_case_cnt
            pd_cnt_threshold = pos_num_th * sr_case_num
            pd_patch_score['cnt_threshold'] = pd_cnt_threshold
            pd_patch_score['predict'] = pd_patch_score['pos_patch_cnt'] - pd_patch_score['cnt_threshold']
            pd_patch_score['predict'] = pd_patch_score['predict'].map(lambda x: 1 if x > 0 else 0)
            pd_patch_score = pd.merge(pd_patch_score, pd_patch_mean_score, on='case')

            # calculate metric scores
            pd_case_score = pd.merge(pd_case_score, pd_Y_case_label, on='case')
            pd_case_input = pd_case_score[['case', 'mean_score', 'predict', 'label']]
            Y_case_predict = np.array(pd_case_input['predict'])
            Y_case_score = np.array(pd_case_input['mean_score'])

            print(f"acc for case prediction {accuracy_score(Y_case_label, Y_case_predict)}")
            test_metrics = metric_func(Y_case_score, Y_case_predict, Y_case_label)

            print("\ncalculate for case prediction")
            time_save_dict, pd_pred_label = calc_time_save(pd_case_input)
            time_save_dict['TIME'] = case_time
            case_pred_record = {**test_metrics, **time_save_dict}
            case_pred_record = {'case_' + k: v for k, v in case_pred_record.items()}

            Y_case_predict = np.array(pd_patch_score['predict'])
            pd_patch_score = pd.merge(pd_patch_score, pd_Y_case_label, on='case')
            pd_patch_input = pd_patch_score[['case', 'mean_score', 'predict', 'label']]
            Y_case_score = np.array(pd_patch_input['mean_score'])
            test_metrics = metric_func(Y_case_score, Y_case_predict, Y_case_label)

            print("\ncalculate for patch prediction")
            time_save_dict, pd_pred_label = calc_time_save(pd_patch_input)
            time_save_dict['TIME'] = patch_time
            patch_pred_record = {**test_metrics, **time_save_dict}
            patch_pred_record = {'patch_' + k: v for k, v in patch_pred_record.items()}

            # calculate for GFLOW - blend of case and patch prediction
            pd_gflow_score = copy.deepcopy(pd_case_score)
            ca_index = pd_gflow_score['predict'] == 1
            pd_gflow_score.loc[~ca_index, 'predict'] = pd_patch_score.loc[~ca_index, 'predict']
            pd_gflow_score.loc[~ca_index, 'mean_score'] = pd_patch_score.loc[~ca_index, 'mean_score']

            if Config.PATCH_ONLY:
                pd_case_score = pd_case_score[~ca_index].reset_index()
                Y_case_label = pd.DataFrame(Y_case_label)[~ca_index]['label']
                ca_names = pd_case_score['case']
                ca_index = pd_case_score['predict'] == 1

            pd_gflow_score['case_lb'] = 'non-CA'
            pd_gflow_score.loc[ca_index, 'case_lb'] = 'CA'
            pd_gflow_input = pd_gflow_score[['case', 'mean_score', 'predict', 'label', 'case_lb']]
            Y_case_predict = np.array(pd_gflow_score['predict'])
            Y_case_score = np.array(pd_gflow_score['mean_score'])
            test_metrics = metric_func(Y_case_score, Y_case_predict, Y_case_label)

            # generate roc_auc curve plot
            if Config.SAVE_ROC_CURVE_PLOT:
                roc_curve_dir = os.path.join(output_dir, 'roc_curve')
                os.makedirs(roc_curve_dir, exist_ok=True)
                roc_curve_plot_path = os.path.join(roc_curve_dir, th_comb_name + '.svg')
                plt.title('GCNet vs. GastroFLOW on External Dataset')
                fpr, tpr, _ = metrics.roc_curve(Y_case_label, np.array(pd_case_score['mean_score']))
                gcnet_roc_auc = metrics.auc(fpr, tpr)
                plt.plot(fpr, tpr, 'r', label='GCNet (AUC = %0.3f)' % gcnet_roc_auc)
                fpr, tpr, _ = metrics.roc_curve(Y_case_label, np.array(pd_gflow_score['mean_score']))
                gflow_roc_auc = metrics.auc(fpr, tpr)
                plt.plot(fpr, tpr, 'b', label='GastroFLOW (AUC = %0.3f)' % gflow_roc_auc)
                plt.legend(loc='lower right')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.ylabel('Sensitivity')
                plt.xlabel('1 - Specificity')
                plt.savefig(roc_curve_plot_path, format="svg")
                plt.close()

                cm_dir = os.path.join(output_dir, 'cm')
                os.makedirs(cm_dir, exist_ok=True)
                cm_plot_path = os.path.join(cm_dir, th_comb_name + '.svg')
                lbs = [1, 0]
                ax = plt.subplot()
                conf_mat = confusion_matrix(Y_case_label, Y_case_predict, labels=lbs)
                conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
                sns.set(font="Arial")
                sns.set(font_scale=1.2)
                sns.heatmap(conf_mat, annot=True, annot_kws={"fontsize": 18}, fmt='.3f', ax=ax, cmap="Blues", vmin=0.0,
                            vmax=1.0)
                cbar = ax.collections[0].colorbar
                cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                label_size = 8
                ax.set_xlabel('Predicted labels', fontsize=label_size)
                ax.set_ylabel('True labels', fontsize=label_size)
                ax.set_title('Normalized Confusion Matrix', fontsize=label_size)
                ax.xaxis.set_ticklabels(['CA', 'non-CA'], fontsize=label_size)
                ax.yaxis.set_ticklabels(['Carcinoma', 'Non-Carcinoma'], fontsize=label_size)
                plt.savefig(cm_plot_path, format='svg')
                plt.close()

            # save prediction scores for nominated thresholds
            if th_comb_name == Config.tgt_thresholds:
                pred_save_dir = os.path.join(output_dir, "pred_score")
                os.makedirs(pred_save_dir, exist_ok=True)
                pred_file = os.path.join(pred_save_dir, 'gcnet_' + th_comb_name + '.xlsx')
                pd_tmp_pred = pd_case_score[['case', 'mean_score', 'label']]
                pd_tmp_pred = pd_tmp_pred.sort_values(by=['label', 'mean_score'], ascending=True).reset_index(
                    drop=True)
                pd_tmp_pred['case'] = pd_tmp_pred['case'].apply(
                    lambda x: x.replace('Stomach ', '').replace(' ', ''))
                pd_tmp_pred.to_excel(pred_file, index=False)
                pred_file = os.path.join(pred_save_dir, 'glow_' + th_comb_name + '.xlsx')
                pd_tmp_pred = pd_gflow_score[['case', 'mean_score', 'predict', 'label', 'case_lb']]
                pd_tmp_pred = pd_tmp_pred.sort_values(by=['label', 'mean_score'], ascending=True).reset_index(
                    drop=True)
                pd_tmp_pred['case'] = pd_tmp_pred['case'].apply(
                    lambda x: x.replace('Stomach ', '').replace(' ', ''))
                pd_tmp_pred.to_excel(pred_file, index=False)
                pd_tmp_pred = pd_tmp_pred.sort_values(by=['case_lb', 'mean_score'], ascending=True).reset_index(
                    drop=True)
                pd_tmp_pred['case'] = pd_tmp_pred['case'].apply(
                    lambda x: x.replace('Stomach ', '').replace(' ', ''))
                pd_tmp_pred.to_excel(pred_file.replace(th_comb_name, 'sorted_' + th_comb_name), index=False)
                pd_tmp_pred = pd_tmp_pred.sort_values(by=['label', 'mean_score'], ascending=False).reset_index(
                    drop=True)
                pd_tmp_pred['case'] = pd_tmp_pred['case'].apply(
                    lambda x: x.replace('Stomach ', '').replace(' ', ''))
                pd_tmp_pred.to_excel(pred_file.replace(th_comb_name, 'sorted_2_' + th_comb_name), index=False)

            if Config.SAVE_TRIAGE:
                triage_dir = os.path.join(output_dir, 'triage_plot/')
                os.makedirs(triage_dir, exist_ok=True)
                triage_path = os.path.join(triage_dir, th_comb_name + '.svg')
            else:
                triage_path = None

            print("\ncalculate for GFLOW prediction")
            time_save_dict, pd_pred_label = calc_time_save(pd_gflow_input, triage_path)
            ca_ratio = np.sum(ca_index) / len(pd_case_score)
            # calculate the theoretical time consumption for GFLOW
            time_save_dict['TIME'] = case_time + (1 - ca_ratio) * patch_time
            gflow_pred_record = {**test_metrics, **time_save_dict}
            gflow_pred_record = {'GFLOW_' + k: v for k, v in gflow_pred_record.items()}

            record_dict = {**case_pred_record, **patch_pred_record, **gflow_pred_record, **record_dict}
            record_dict_lst.append(record_dict)

        pd_all_record = pd.DataFrame(record_dict_lst)

        # generate thermal plot, just for reference
        if Config.GEN_THERMAL:
            thermal_plot_path = os.path.join(output_dir, 'threshold_plot.svg')
            metric_list = ['GFLOW_NPV', 'GFLOW_TPR', 'GFLOW_MCC']
            pd_all_record['threshold_metric'] = pd_all_record.loc[:, metric_list].apply(sum, axis=1)

            y_arr = pd_all_record['threshold_metric'].to_numpy().reshape((len(th1_lst), len(th3_lst)))
            y_arr = np.flip(y_arr, axis=0)
            plt.figure()
            plt.imshow(y_arr)
            plt.xlabel('pos_num_th')
            plt.xticks(list(range(len(th3_lst))), th3_lst)
            plt.ylabel('md_score_th')
            plt.yticks(list(range(len(th1_lst))), th1_lst)

            TH1, TH3 = np.meshgrid(th1_lst, th3_lst)
            score = pd_all_record['threshold_metric'].to_numpy().reshape((len(th1_lst), len(th3_lst)))
            fig = plt.figure()
            ax = fig.gca(projection="3d", title="Metric score with thresholds")
            ax.plot_surface(TH3, TH1, np.exp(score), color="blue", linestyle="-")
            ax.set_xlabel("pos_num_th")
            ax.set_ylabel("md_score_th")
            ax.set_zlabel("metric_score")
            plt.show()

            plt.savefig(thermal_plot_path, format="svg")
            plt.close()

        # save all metric scores
        pd_all_record.to_excel(output_dir + f"result_record.xlsx", index=False)
