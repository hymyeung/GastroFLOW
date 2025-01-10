import os
import re
from glob import glob

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from Tool import find_case
from Tool import keep_digits


# Config setting
class Config:
    train_dataset = 'sample_data'
    feature_base_dir = f"../dataset/feature_dataset/{train_dataset}/"
    cellular_feat_dir = f"../dataset/feature_dataset/{train_dataset}/cellular_feature/"
    csv_path = f"../dataset/feature_dataset/{train_dataset}/aggregated_data.csv"
    patch_size_lst = [500]
    patch_saving_dir = f"../dataset/feature_dataset/{train_dataset}/patch/"
    label_csv_path = f"../dataset/feature_dataset/{train_dataset}/case_label.csv"
    case_match_pattern = '(\d+)AH(\d+\-*[A-Z]*)-H&E'
    rm_outlier = False


if __name__ == "__main__":

    # Build label mapping dict
    pd_label = pd.read_csv(Config.label_csv_path)
    cases = pd_label['case'].tolist()
    labels = pd_label['label'].tolist()
    ndpi_files = pd_label['ndpi_files'].tolist()
    # ndpi_files = [eval(r) for r in ndpi_files]
    ndpi_files = [os.path.basename(f).split('.ndpi')[0] for f in ndpi_files]
    case_lb_dict = {}
    lb_name_dict = {0: 'N', 1: 'C'}
    for case, lb in zip(cases, labels):
        case_lb_dict[case] = lb_name_dict[lb]

    # Data aggregation for WSI
    pd_aggregate = pd.DataFrame()
    key_columns = ['id', 'Nucleus: Area', 'Nucleus: Perimeter', 'Nucleus: Circularity', 'Nucleus: Max caliper',
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
    drop_columns = ['Image', 'Name', 'Class', 'Parent', 'ROI', 'Centroid X Âµm', 'Centroid Y Âµm']
    substitute_column_pairs = [['Centroid X px', 'Centroid X µm'], ['Centroid Y px', 'Centroid Y µm']]
    pixel_length_micron = 0.22096517588828002
    all_txt_files = glob(Config.cellular_feat_dir + '*.txt')
    txt_base_files = [os.path.basename(f) for f in all_txt_files]
    mode_lst = ['pos_neg', 'pos', 'neg']
    POS_NEG_MODE = mode_lst[0]
    if 'PolyU' in Config.train_dataset:
        ndpi_filt = True
    else:
        ndpi_filt = False
    for file in tqdm(all_txt_files, desc='Extracting feature for WSI ...'):
        # if ndpi_filt and (os.path.basename(file).split('.')[0] not in ndpi_files):
        #     continue
        base_name = os.path.basename(file)
        df_tmp = pd.read_csv(file, sep='\t', encoding='ISO-8859-1')
        df_tmp.rename(columns={"Class": "Classification"}, inplace=True)
        if POS_NEG_MODE == 'pos':
            df_tmp = df_tmp[df_tmp['Classification'] == 'Positive']
        if POS_NEG_MODE == 'neg':
            df_tmp = df_tmp[df_tmp['Classification'] == 'Negative']
        df_tmp.insert(0, 'id', base_name)
        df_tmp = df_tmp.loc[:, key_columns]
        # filter outliers
        if Config.rm_outlier:
            df_tmp = df_tmp[(np.abs(stats.zscore(df_tmp.loc[:, key_columns[1:]])) < 3).all(axis=1)]
        ids = pd.Series({'id': base_name})
        mean_of_df = df_tmp.mean(numeric_only=True)
        df_combined = pd.DataFrame(pd.concat([ids, mean_of_df], axis=0)).transpose()
        pd_aggregate = pd.concat([pd_aggregate, df_combined], axis=0, ignore_index=True)
    pd_aggregate['case'] = pd_aggregate['id'].map(find_case)
    pd_aggregate['Import_diagnosis'] = pd_aggregate['case'].map(case_lb_dict)
    pd_aggregate.sort_values(by='case', inplace=True)
    pd_aggregate.to_csv(Config.csv_path, index=False)
    pd_aggregate.to_excel(Config.csv_path[:-4] + '.xlsx', index=False)

    # Data aggregation for tiles
    patch_size = 0
    for patch_size in Config.patch_size_lst:
        pd_aggregate_patch = pd.DataFrame([])
        for file in tqdm(all_txt_files, desc='Extracting feature for patches ...'):
            base_name = os.path.basename(file)
            df_tmp = pd.read_csv(file, sep='\t', header=0)
            df_tmp.rename(columns={"Class": "Classification"}, inplace=True)
            if POS_NEG_MODE == 'pos':
                df_tmp = df_tmp[df_tmp['Classification'] == 'Positive']
            if POS_NEG_MODE == 'neg':
                df_tmp = df_tmp[df_tmp['Classification'] == 'Negative']
            for substitute_column_pair in substitute_column_pairs:
                if substitute_column_pair[0] in df_tmp.columns:
                    df_tmp[substitute_column_pair[0]] = df_tmp[substitute_column_pair[0]] * pixel_length_micron
                    df_tmp.rename(columns={substitute_column_pair[0]: substitute_column_pair[1]}, inplace=True)
            df_tmp.insert(loc=7, column="X", value=np.floor(df_tmp['Centroid X µm'].values / (patch_size / 4)))
            df_tmp.insert(loc=8, column="Y", value=np.floor(df_tmp['Centroid Y µm'].values / (patch_size / 4)))
            df_patch = df_tmp.groupby(['X', 'Y']).mean(numeric_only=True)
            df_patch['cell_count'] = df_tmp.groupby(['X', 'Y']).size()
            df_patch.reset_index(inplace=True)
            df_patch.insert(loc=0, column='id', value=base_name)
            df_patch = df_patch[df_patch.cell_count > 10]

            output_dir = os.path.join(Config.patch_saving_dir, str(patch_size))
            # os.makedirs(output_dir, exist_ok=True)
            # csv_file = os.path.join(output_dir, f"{base_name}_tile_data.csv")
            # df_patch.to_csv(csv_file, index=False)

            pd_aggregate_patch = pd.concat([pd_aggregate_patch, df_patch], axis=0)

        pd_aggregate_patch = pd_aggregate_patch.loc[:, key_columns]
        pd_aggregate_patch['case'] = pd_aggregate_patch['id'].map(find_case)
        pd_aggregate_patch['Import_diagnosis'] = pd_aggregate_patch['case'].map(case_lb_dict)
        pd_aggregate_patch.sort_values(by='case', inplace=True)
        output_csv_file = os.path.join(Config.feature_base_dir, f'aggregated_patch_data_{patch_size}.csv')
        pd_aggregate_patch.to_csv(output_csv_file, index=False)
        output_xlsx_file = os.path.join(Config.feature_base_dir, f'aggregated_patch_data_{patch_size}.xlsx')
        pd_aggregate_patch.to_excel(output_xlsx_file, index=False)
