#!/usr/bin/env python
# coding: utf-8

import os
import pickle
from glob import glob

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from Tool import metric_func

if __name__ == "__main__":
    # loading dataset
    dataset_lst = ['sample_data']
    train_dataset = dataset_lst[0]
    test_dataset = dataset_lst[0]
    key_features = ['Nucleus: Area', 'Nucleus: Perimeter', 'Nucleus: Circularity', 'Nucleus: Max caliper',
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
                    'Cytoplasm: Eosin OD min', 'Nucleus/Cell area ratio']
    label_column = 'Import_diagnosis'
    lb_dict = {'N': 0, 'C': 1}
    external_cellular_feature_path = f"../dataset/feature_dataset/{test_dataset}/aggregated_data.csv"
    output_dir = f"../result/{test_dataset}/machine_learning/infer/"
    os.makedirs(output_dir, exist_ok=True)
    pd_data = pd.read_csv(external_cellular_feature_path)
    all_cases = pd_data['case']
    data_ids = pd_data['id'].values
    X_dataset = pd_data.loc[:, key_features].values
    Y_dataset = pd_data.loc[:, label_column].map(lb_dict).values

    # preparing models
    random_seed = 42
    np.random.seed(random_seed)
    algorithm_settings = {'LR': {'LR_lbfgs': LogisticRegression(solver='lbfgs'),
                                 'LR_sag': LogisticRegression(solver='sag'),
                                 'LR_saga': LogisticRegression(solver='saga'),
                                 'LR_liblinear': LogisticRegression(solver='liblinear'),
                                 'LR_newton-cg': LogisticRegression(solver='newton-cg')},
                          'KNN': {'KNN_manhattan': KNeighborsClassifier(n_neighbors=3, metric='manhattan', p=2),
                                  'KNN_chebyshev': KNeighborsClassifier(n_neighbors=3, metric='chebyshev', p=2),
                                  'KNN_minkowski': KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2)},
                          'SVC': {'SVC_linear': SVC(kernel='linear', gamma='scale'),
                                  'SVC_poly': SVC(kernel='poly', gamma='scale'),
                                  'SVC_rbf': SVC(kernel='rbf', gamma='scale'),
                                  'SVC_sigmoid': SVC(kernel='sigmoid', gamma='scale')},
                          'GB': {'GB': GaussianNB()},
                          'DT': {'DT_entropy': DecisionTreeClassifier(criterion='entropy'),
                                 'DT_gini': DecisionTreeClassifier(criterion='gini')},
                          'RF': {'RF_entropy_10': RandomForestClassifier(criterion='entropy', n_estimators=10),
                                 'RF_entropy_100': RandomForestClassifier(criterion='entropy', n_estimators=100),
                                 'RF_gini_10': RandomForestClassifier(criterion='gini', n_estimators=10),
                                 'RF_gini_100': RandomForestClassifier(criterion='gini', n_estimators=100)}}
    model_sav_dir = f"../result/{train_dataset}/machine_learning/by_image/ml_model/"
    all_model_files = glob(model_sav_dir + '*.sav')

    # load scaler
    scaler_model_file = [file for file in all_model_files if 'scaler' in file][0]
    sc = pickle.load(open(scaler_model_file, 'rb'))
    X_dataset = sc.transform(X_dataset)

    # filter out target algorithms
    target_algors = ['SVC_linear', 'LR_lbfgs', 'LR_liblinear']
    all_model_files = [f for f in all_model_files if '_'.join(os.path.basename(f)[:-4].split('_')[1:3]) in target_algors]

    # inferring prediction
    model_pred_lst = []
    for model_name, settings in algorithm_settings.items():
        for model_parameter, _ in settings.items():
            model_sav_lst = [file for file in all_model_files if model_parameter in file]
            if len(model_sav_lst) == 0:
                continue
            score_lst = []
            for model_sav in model_sav_lst:
                print(f"using {os.path.basename(model_sav).split('.')[0]} to classify dataset...")
                classifier = pickle.load(open(model_sav, 'rb'))
                Y_pred = classifier.predict(X_dataset)
                score_rst = metric_func(Y_pred, Y_pred, Y_dataset)
                score_lst.append(list(score_rst.values()))
            avg_score = np.mean(np.array(score_lst), axis=0)
            avg_score = {col: score for (col, score) in zip(score_rst.keys(), avg_score)}
            score_dict = dict()
            score_dict['model'] = model_parameter
            score_dict.update(avg_score)
            model_pred_lst.append(score_dict)

    # calculate metrics
    pd_all_score = pd.DataFrame(model_pred_lst).sort_values(by='model')
    pd_all_score.to_csv(output_dir + "score.csv", index=False)
    pd_avg_all_score = pd_all_score.groupby(by='model').mean()
    pd_avg_all_score = pd_avg_all_score.reset_index()
    all_method_avg_score = pd_avg_all_score.mean(numeric_only=True)
    all_method_avg_score['model'] = 'all_model_avg'
    pd_avg_all_score = pd.concat([pd_avg_all_score, pd.DataFrame(all_method_avg_score).transpose()],
                                 axis=0).reset_index(
        drop=True)
    pd_avg_all_score.to_excel(output_dir + "avg_score.xlsx", index=False)

    print(pd_avg_all_score)
