# Python3

import numpy as np
from scipy import stats

def get_avg_pix_on_mat_area (mat, i1 = (0, 0), i2 = (0, 0)) :
    if i1 == i2 :
     return [np.average (mat[:, :, i]) for i in (0, 1, 2)]
    return [np.average(mat[i1[0] : i1[1], i2[0] : i2[1], i3]) for i3 in (0, 1, 2)]

def dif_of_pixels (pix1, pix2) :
    return np.array(pix1) - np.array(pix2)

def get_confidence_intervals (mat, i1 = (0, 0), i2 = (0, 0)) :
    if i1 == i2:
        i1 = (0, mat.shape[0] - 1)
        i2 = (0, mat.shape[1] - 1)
    a = [mat[i1[0] : i2[0], i1[1] : i2[1], i] for i in (0, 1, 2)]
    mean = [np.mean (a[i]) for i in (0, 1, 2)]
    print(mean)
    std = [np.std (a[i]) for i in (0, 1, 2)]
    print(std)
    return [stats.norm.interval (0.4, mean[i], std[i]) for i in (0, 1, 2)]

def find_low_high (mat, i1 = (0, 0), i2 = (0, 0)):
    a = [mat[i1[0]: i2[0], i1[1]: i2[1], i] for i in (0, 1, 2)]
    return np.array ([np.min(a[i]) for i in (0, 1, 2)]), np.array ([np.max(a[i]) for i in (0, 1, 2)])

def find_avg_low_high (mat, i1 = (0, 0), i2 = (0, 0)):
    rows = i2[0] - i1[0]
    cols = i2[1] - i1[1]

    row_step = rows // 3
    if (row_step == 0):
        row_step = 1
    col_step = cols // 3
    if (col_step == 0):
        col_step = 1

    mean_arr = np.repeat (np.array ([[0, 0, 0]]), 9, axis = 0)
    for i in (0, 1, 2):
        for j in (0, 1, 2):
            mean_arr[i * 3 + j] = [np.mean (mat[i1[0] + i * row_step: i1[0] + (i + 1) * row_step, \
                                    i2[0] + j * col_step: i2[0] + (j + 1) * col_step, \
                                    k]) for k in (0, 1, 2)]
    #print (mean_arr)
    return np.array (np.min (mean_arr, axis = 0)), np.array (np.max (mean_arr, axis = 0))

    #print (mean_arr)

def modify_image (mat, conf_ints):
    def check_pix (pix, conf_ints) -> bool:
        for i in (0, 1, 2):
            if (pix[i] <= conf_ints[i][0] or pix[i] >= conf_ints[i][1]):
                return False
        return True

    rows, cols, chanels = mat.shape
    for i in range (3, rows - 4):
        for j in range (3, cols - 4):
            pix = [np.mean(mat[i - 3: i + 3, j - 3: j + 3, k]) for k in (0, 1, 2)]
            if (check_pix (pix, conf_ints) == False):
                mat[i, j] = [0, 0, 0]
            else:
                mat[i, j] = [255, 255, 255]
