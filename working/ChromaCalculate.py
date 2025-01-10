import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import linalg

Image.MAX_IMAGE_PIXELS = None


# pick out common elements in multiple lists
def common_member(a, b, c):
    a_set = set(a)
    b_set = set(b)
    c_set = set(c)
    return list(a_set & b_set & c_set)


# calculate source stimulus based on IAM-9C-00446
def generate_chromatic_stimulus_value():
    transmit_file = "../dataset/color_board_dataset/IAM-9C-00446.xlsm"
    pd_tm = pd.read_excel(transmit_file)
    pd_tm = pd_tm.iloc[:99, :26]
    light_source_file = "../dataset/color_board_dataset/CIE_std_illum_D65.csv"
    pd_ls = pd.read_csv(light_source_file, names=['Wavelength', 'Value'])
    observer_data_file = "../dataset/color_board_dataset/CIE_xyz_1931_2deg.csv"
    pd_ob = pd.read_csv(observer_data_file, names=['Wavelength', 'X', 'Y', 'Z'])
    wavelength_range = common_member(pd_tm['Wavelength'], pd_ls['Wavelength'], pd_ob['Wavelength'])
    color_names = list(pd_tm.columns)[1:]
    chrome_stimulus_dict = dict()
    xyz2lrgb_mat = np.array([[3.2410, -1.5374, -0.4986], [-0.9692, 1.8760, 0.0416], [0.0556, -0.2040, 1.0570]])
    for color in color_names:
        transmit_arr = np.expand_dims(np.array(pd_tm[color][pd_tm['Wavelength'].isin(wavelength_range)]), axis=-1)
        light_source_arr = np.expand_dims(np.array(pd_ls.loc[pd_ls['Wavelength'].isin(wavelength_range), 'Value']),
                                          axis=-1)
        observer_arr = np.array(
            pd_ob.loc[pd_ob['Wavelength'].isin(wavelength_range), ['X', 'Y', 'Z']])
        chromatic_coordinates = np.mean(light_source_arr * observer_arr * transmit_arr, axis=0)
        chromatic_coordinates = chromatic_coordinates / np.sum(chromatic_coordinates)
        chrome_stimulus_dict[color] = xyz2lrgb_mat @ chromatic_coordinates
    chromatic_stimulus_file = "../dataset/color_board_dataset/chromatic_stimulus_file.csv"
    pd.DataFrame(chrome_stimulus_dict).to_csv(chromatic_stimulus_file)


# transform RGB value into lRGB
def transform_to_lrgb_val(rgb_val):
    lrgb_val = rgb_val / 255
    threshold = 0.04045
    if lrgb_val <= threshold:
        lrgb_val = lrgb_val / 12.92
    else:
        lrgb_val = np.power((lrgb_val + 0.055) / 1.055, 2.4)
    return lrgb_val


# transform image array from RGB domain into lRGB domain
def transform_to_lrgb_arr(rgb_arr):
    lrgb_arr = rgb_arr.astype(np.float32) / 255
    threshold = 0.04045
    up_part = np.where(lrgb_arr > threshold)
    dn_part = np.where(lrgb_arr <= threshold)
    lrgb_arr[up_part] = np.power((lrgb_arr[up_part] + 0.055) / 1.055, 2.4)
    lrgb_arr[dn_part] = lrgb_arr[dn_part] / 12.92
    return lrgb_arr


# restore lRGB value to RGB value
def rgb_inverse_transform(lrgb_val):
    threshold = 0.0031308
    if lrgb_val <= threshold:
        rgb_val = 12.92 * lrgb_val
    else:
        rgb_val = 1.055 * np.power(lrgb_val, 1 / 2.4) - 0.055
    rgb_val = np.clip(rgb_val, 0, 1) * 255
    return rgb_val


# transform image arr from lRGB domain back into RGB domain
def rgb_arr_inverse_transform(lrgb_arr):
    threshold = 0.0031308
    up_part = np.where(lrgb_arr > threshold)
    dn_part = np.where(lrgb_arr <= threshold)
    lrgb_arr[up_part] = 1.055 * np.power(lrgb_arr[up_part], 1 / 2.4) - 0.055
    lrgb_arr[dn_part] = 12.92 * lrgb_arr[dn_part]
    rgb_arr = (np.clip(lrgb_arr, 0, 1) * 255).astype(np.uint8)
    return rgb_arr


# solve matrix equation
def get_ls_solution(mat_a, mat_b):
    mat_a_pinv = linalg.pinv(mat_a)
    p = mat_a_pinv @ mat_b
    return p


# color normalization function - project image array into target color domain
def color_normalize(data_arr, trans_mat):
    data_shape = data_arr.shape
    if trans_mat.shape[0] == 4:
        data_arr = transform_to_lrgb_arr(data_arr)
        data_arr = (np.concatenate([np.ones((data_shape[:2] + (1,))), data_arr], axis=-1) @ trans_mat)[:, :, 1:]
        data_arr = rgb_arr_inverse_transform(data_arr)
    elif trans_mat.shape[0] == 3:
        data_arr = transform_to_lrgb_arr(data_arr)
        data_arr = data_arr @ trans_mat
        data_arr = rgb_arr_inverse_transform(data_arr)
    return data_arr


if __name__ == "__main__":
    # calculate stimulus values
    generate_chromatic_stimulus_value()

    # load all input dataset and target dataset
    chromatic_stimulus_file = "../dataset/color_board_dataset/chromatic_stimulus_file.csv"
    pd_xyz = pd.read_csv(chromatic_stimulus_file)
    polyu_color_board_data_file = "../dataset/color_board_dataset/PolyU/avg_rgb.csv"
    pd_polyu_color_data = pd.read_csv(polyu_color_board_data_file)
    scn400_color_board_data_file = "../dataset/color_board_dataset/SCN400/avg_rgb.csv"
    pd_scn400_color_data = pd.read_csv(scn400_color_board_data_file)

    # choose reference color pads
    color_pack_names = [m + n for m in ['A', 'B', 'C', 'D'] for n in ['1', '2', '3', '4', '5', '6']]
    # exclude_names = ['B3', 'B4', 'B5', 'C3', 'C4', 'C5']
    exclude_names = []
    color_pack_names = [c for c in color_pack_names if c not in exclude_names]
    print(len(color_pack_names))

    # dataset pre-processing
    polyu_data_arr = np.array(pd_polyu_color_data.loc[:, color_pack_names].applymap(transform_to_lrgb_val))
    add_constant = False
    if add_constant:
        polyu_data_arr = np.concatenate([np.ones((1, polyu_data_arr.shape[1])), polyu_data_arr])
    polyu_data_arr = polyu_data_arr.transpose()
    scn400_data_arr = np.array(pd_scn400_color_data.loc[:, color_pack_names].applymap(transform_to_lrgb_val))
    if add_constant:
        scn400_data_arr = np.concatenate([np.ones((1, scn400_data_arr.shape[1])), scn400_data_arr])

    # solve color transformation matrix
    scn400_data_arr = scn400_data_arr.transpose()
    xyz_arr = np.array(pd_xyz.loc[:, color_pack_names]).transpose()
    scn400_sol_arr = get_ls_solution(scn400_data_arr, xyz_arr)
    polyu_sol_arr = get_ls_solution(polyu_data_arr, xyz_arr)
    pts_trans_mat = polyu_sol_arr @ linalg.pinv(scn400_sol_arr)
    pts_trans_mat_d = get_ls_solution(polyu_data_arr, scn400_data_arr)
    transform_mat_file = "../dataset/color_board_dataset/polyu2scn400_trans_mat_d65.npy"
    np.save(transform_mat_file, pts_trans_mat)

    # try transformation matrix and display transformation results
    # sample_files = glob.glob("../dataset/color_board_dataset/PolyU/Color Pack Sample/*.tif")
    sample_files = glob.glob("../dataset/patch_dataset/20231010-PolyU-PYNEH_slide/20231010-PolyU-PYNEH_slide/*.png")
    sample_files = sorted(sample_files)
    scn_dir = "../dataset/color_board_dataset/SCN400/Color Pack Sample/"
    for sample_file in sample_files:
        # sample_file = "./sample.png"
        samp_arr = np.array(Image.open(sample_file))
        rec_samp_arr_1 = color_normalize(samp_arr, pts_trans_mat)
        rec_samp_arr_2 = color_normalize(samp_arr, pts_trans_mat_d)
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(samp_arr)
        plt.subplot(2, 2, 2)
        plt.imshow(rec_samp_arr_1)
        plt.subplot(2, 2, 3)
        plt.imshow(rec_samp_arr_2)
        scn_file = os.path.join(scn_dir, os.path.basename(sample_file))
        if os.path.exists(scn_file):
            scn_arr = np.array(Image.open(scn_file))
            plt.subplot(2, 2, 4)
            plt.imshow(scn_arr)
        plt.show()

