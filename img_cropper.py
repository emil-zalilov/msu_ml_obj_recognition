import numpy as np
import cv2 as cv

empty_img = np.empty ((ROWS, COLS, 3), np.uint8)

r_mid = ROWS // 2
c_mid = COLS // 2

def get_img (img_t):
    r, c, zzz = img_t.shape
    ## reshaping to (rows, cols) size
    img = empty_img.copy ()

    r_m = r // 2
    if r < ROWS:
        delta_r = r_m
    else:
        delta_r = r_mid

    c_m = c // 2
    if c < COLS:
        delta_c = c_m
    else:
        delta_c = c_mid

    img[r_mid - delta_r: r_mid + delta_r, c_mid - delta_c: c_mid + delta_c] = \
            img_t[r_m - delta_r : r_m + delta_r, c_m - delta_c : c_m + delta_c]

    return img