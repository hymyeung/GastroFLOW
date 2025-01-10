import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from Tool import metric_func


if __name__ == "__main__":

    random_seed = 42
    np.random.seed(random_seed)

    # set dataset to use
    dataset_lst = ['sample_data']
    train_dataset = dataset_lst[0]

    # set dataset type to use
    PATCH_DATA = False
    cellular_feature_path = f"../dataset/feature_dataset/{train_dataset}/"
    output_dir = f"../result/{train_dataset}/machine_learning/"
    if PATCH_DATA:
        cellular_feature_path = os.path.join(cellular_feature_path, 'aggregated_patch_data_500.csv')
        output_dir = os.path.join(output_dir, 'by_patches')
    else:
        cellular_feature_path = os.path.join(cellular_feature_path, 'aggregated_data.csv')
        output_dir = os.path.join(output_dir, 'by_image')
    result_dir = os.path.join(output_dir, 'train')
    os.makedirs(result_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, 'ml_model')
    os.makedirs(model_dir, exist_ok=True)

    # preprocess dataset
    pd_dataset = pd.read_csv(cellular_feature_path)
    all_cases = pd_dataset['case']
    cross_val_num = 10
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
    all_score_lst = []
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
    MD_SAVE = True
    INTERMEDIATE_SAVE = False

    # data normalization
    X_dataset = pd_dataset.loc[:, key_features].values
    sc = StandardScaler()
    sc.fit(X_dataset)
    model_file = os.path.join(model_dir, f"scaler.sav")
    pickle.dump(sc, open(model_file, 'wb'))

    gkf = GroupKFold(n_splits=cross_val_num)
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X=pd_dataset, groups=all_cases)):
        pd_train = pd_dataset.iloc[train_idx]
        pd_val = pd_dataset.iloc[val_idx]
        train_ids = pd_train['id'].values
        val_ids = pd_val['id'].values
        X_train = pd_train.loc[:, key_features].values
        Y_train = pd_train.loc[:, label_column].map(lb_dict).values
        X_val = pd_val.loc[:, key_features].values
        Y_val = pd_val.loc[:, label_column].map(lb_dict).values

        # data normalization
        X_train = sc.transform(X_train)
        X_val = sc.transform(X_val)

        output_properties = ["id", "gt", "pred"]
        for model_name, settings in algorithm_settings.items():
            for model_parameter, classifier in settings.items():
                classifier.fit(X_train, Y_train)
                Y_pred = classifier.predict(X_val)
                if INTERMEDIATE_SAVE:
                    intermediate_save_dir = os.path.join(result_dir, 'intermediate')
                    os.makedirs(intermediate_save_dir, exist_ok=True)
                    output = np.array([val_ids, Y_val, Y_pred]).transpose()
                    pd_output = pd.DataFrame(columns=output_properties, data=output)
                    pd_output.to_csv(intermediate_save_dir + f"fold{fold}_{model_parameter}.csv")

                score_dict = dict()
                score_dict['fold'] = fold
                score_dict['model'] = model_parameter
                Y_pred_lb = (Y_pred > 0.5).astype(int)
                score_dict.update(metric_func(Y_pred, Y_pred_lb, Y_val))
                all_score_lst.append(score_dict)

                if MD_SAVE:
                    model_file = os.path.join(model_dir, f"fold{fold}_{model_parameter}.sav")
                    pickle.dump(classifier, open(model_file, 'wb'))

    # calculate metric score and save
    pd_all_score = pd.DataFrame(all_score_lst).sort_values(by=['model', 'fold'], ascending=[True, True])
    pd_all_score.to_csv(os.path.join(result_dir, 'score.csv'), index=False)
    pd_avg_all_score = pd_all_score.groupby(by='model').mean()
    pd_avg_all_score = pd_avg_all_score.reset_index().drop('fold', axis=1)
    all_method_avg_score = pd_avg_all_score.mean(numeric_only=True)
    all_method_avg_score['model'] = 'all_model_avg'
    pd_avg_all_score = pd.concat([pd_avg_all_score, pd.DataFrame(all_method_avg_score).transpose()],
                                 axis=0).reset_index(drop=True)
    pd_avg_all_score.to_excel(os.path.join(result_dir, 'avg_score.xlsx'), index=False)

    print(pd_avg_all_score)
