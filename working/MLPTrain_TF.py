# -*- coding: utf-8 -*-

from tensorflow import keras
import numpy as np
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from Tool import find_case, metric_func
import math

import numbers
import os
import six

import numpy
import matplotlib.collections
from matplotlib import pyplot
from sklearn import metrics
from sklearn.metrics import auc


class Config:
    dataset_lst = [
        "sample_data",
        "sample_data",
        "sample_data",
    ]
    train_data_file_lst = [f"../dataset/feature_dataset/{d}/aggregated_data.csv" for d in dataset_lst]
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
    val_data_ratio = 0.2
    batch_size = 40
    epoch_num = 20
    train_dataset_name = 'sample_data'
    ckpt_dir = f"../ckpt/tensorflow/{train_dataset_name}/"
    output_dir = f"../result/tensorflow/{train_dataset_name}/"


if __name__ == "__main__":
    random_seed = 42
    np.random.seed(random_seed)

    # load training dataset
    assert len(Config.train_data_file_lst) > 0
    data_lst = []
    for data_file in Config.train_data_file_lst:
        df_tmp = pd.read_csv(data_file)
        data_lst.append(df_tmp)
    df_data = pd.concat(data_lst, axis=0).reset_index(drop=True)
    df_data['case'] = df_data['id'].map(find_case)
    X_dataset = df_data.loc[:, Config.key_features].values
    Y_dataset = df_data.loc[:, Config.label_column].map(Config.lb_dict).values
    case_names = df_data['case'].tolist()

    X_train, X_val, Y_train, Y_val = train_test_split(X_dataset, Y_dataset, test_size=Config.val_data_ratio,
                                                      random_state=random_seed, stratify=case_names)

    # preprocess data
    sc = StandardScaler()
    sc.fit(X_dataset)
    X_train = sc.transform(X_train)
    X_val = sc.transform(X_val)

    # define a simple MLP model
    model = keras.Sequential()
    model.add(layers.Dense(95, input_dim=41, activation="relu", activity_regularizer=regularizers.l2(0.35)))
    model.add(layers.Dropout(0.05))
    model.add(layers.Dense(95, activation="relu"))
    model.add(layers.Dropout(0.15))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='Nadam', loss='logcosh', metrics=['accuracy'])

    X_train = K.cast_to_floatx(X_train)
    Y_train = K.cast_to_floatx(Y_train)
    X_val = K.cast_to_floatx(X_val)
    Y_val = K.cast_to_floatx(Y_val)

    history = model.fit(X_train, Y_train, batch_size=40, epochs=20, validation_data=(X_val, Y_val))

    Y_pred_score = model.predict(X_val)
    Y_pred_lb = (Y_pred_score > 0.5)

    test_metrics = metric_func(Y_pred_score, Y_pred_lb, Y_val)
    acc = test_metrics['ACC']

    cm = confusion_matrix(Y_val, Y_pred_lb)
    print(cm)

    analysis = [test_metrics['TPR'], test_metrics['TNR'], test_metrics['PPV'], test_metrics['NPV'], test_metrics['ACC'],
                test_metrics['ROC_AUC'], history.history['val_accuracy'][-1]]
    file_name = ', '.join(map(str, analysis))
    os.makedirs(Config.ckpt_dir, exist_ok=True)
    if acc > .85 and history.history['val_accuracy'][-1] > .85:
        model.save(os.path.join(Config.ckpt_dir, file_name+'.h5'))

    os.makedirs(Config.output_dir, exist_ok=True)
    pd.DataFrame(test_metrics, index=[0,]).to_csv(os.path.join(Config.output_dir, file_name+'.csv'), index=False)
