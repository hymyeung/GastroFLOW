import copy
import gc
import os
import shutil
import warnings
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import openslide
from PIL import Image
from tifffile import imwrite
from tqdm import tqdm

from ChromaCalculate import color_normalize

Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # Color normalization patch by patch, otherwise prone to OOM
    PATCH_MODE = True
    PIC_SHOW = False

    # set up input dataset and output directory, normal WSI file include .ndpi, .tif and .svs, etc.
    dataset_name = "sample_data"
    input_ndpi_dir = f"../dataset/tif_dataset/{dataset_name}/"
    if PATCH_MODE:
        output_tif_dir = f"../dataset/tif_dataset/{dataset_name}_patch_normed_d65/"
    else:
        output_tif_dir = f"../dataset/tif_dataset/{dataset_name}_tif_normed_d65/"

    # shutil.rmtree(output_tif_dir)
    os.makedirs(output_tif_dir, exist_ok=True)
    patch_size = 512
    all_tif_path = glob(input_ndpi_dir + '*.tif')
    files_in_output_dir = glob(output_tif_dir + '*.tif')

    trans_mat_file = "../dataset/color_board_dataset/polyu2scn400_trans_mat_d65.npy"
    trans_mat = np.load(trans_mat_file)
    print(trans_mat.shape)

    # start to do color normalization
    for n, file_path in tqdm(enumerate(all_tif_path), total=len(all_tif_path), desc="main progress"):
        base_name = os.path.basename(file_path).split('.')[0]
        print(base_name)
        save_path = os.path.join(output_tif_dir, base_name + '.tif')
        if os.path.exists(save_path):
            continue

        if file_path.endswith('.tif'):
            data_arr = tifffile.imread(file_path)
        elif file_path.endswith('.ndpi'):
            ndpi_img = openslide.OpenSlide(file_path)
            wsi_dims = ndpi_img.dimensions
            data_arr = np.array(ndpi_img.read_region((0, 0), 0, (wsi_dims[0], wsi_dims[1])))[:, :, :3]
        arr_size = data_arr.shape
        print(arr_size)

        if PATCH_MODE:
            # norm_tif_arr = copy.deepcopy(tif_arr)
            norm_tif_arr = np.zeros(arr_size, dtype=np.uint8)
            row_num, col_num = arr_size[0], arr_size[1]
            row_iter_num = arr_size[0] // patch_size
            col_iter_num = arr_size[1] // patch_size
            for i in tqdm(range(row_iter_num)):
                row_start = i * patch_size
                row_end = min(row_num, (i + 1) * patch_size)
                for j in range(col_iter_num):
                    col_start = j * patch_size
                    col_end = min(col_num, (j + 1) * patch_size)
                    patch_arr = data_arr[row_start:row_end, col_start:col_end]
                    if patch_arr.size == 0:
                        continue
                    norm_patch_arr = color_normalize(patch_arr, trans_mat)
                    norm_tif_arr[row_start:row_end, col_start:col_end] = norm_patch_arr

                    # set condition to avoid showing background
                    if PIC_SHOW and (np.nanmean(data_arr[row_start:row_end, col_start:col_end]) < 200):
                        plt.figure()
                        plt.subplot(1, 2, 1)
                        plt.imshow(data_arr[row_start:row_end, col_start:col_end])
                        plt.subplot(1, 2, 2)
                        plt.imshow(norm_tif_arr[row_start:row_end, col_start:col_end])
                        plt.show()
        else:
            norm_tif_arr = color_normalize(data_arr, trans_mat)

        tifffile.imwrite(save_path, data=norm_tif_arr)
