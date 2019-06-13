import numpy as np
import cv2 as cv
import itertools

### settings and methods for hog
bin_n = 9 # Number of bins
affine_flags = cv.WARP_INVERSE_MAP|cv.INTER_LINEAR
cell_size = 8


# methods
def deskew (img):
    m = cv.moments(np.array (img[:,:]))
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    rows, cols = img.shape
    SZ = min (rows, cols)
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img

def hog (img):
    n, m, c = img.shape

    # gradients
    gx = cv.Sobel (img, cv.CV_32F, 1, 0, ksize = 1)
    gy = cv.Sobel (img, cv.CV_32F, 0, 1, ksize = 1)

    mag, ang = cv.cartToPolar (gx, gy, angleInDegrees = True)

    # main values highliting
    indices = np.argmax (mag, axis = 2).reshape ((n, m))
    mag = np.amax (mag, axis = 2).reshape ((n, m))
    for i, j in itertools.product (range (n), range (m)):
        ang[i, j] = ang[i, j, indices[i, j]] % 180

    # cell division
    k_i = n // cell_size
    k_j = m // cell_size

    bin_cells = [ang[0 : cell_size, 0 : cell_size, 0]] * (k_i - 1) * (k_j - 1)
    mag_cells = [mag[0 : cell_size, 0 : cell_size]] * (k_i - 1) * (k_j - 1)
    index = 0
    for i, j in itertools.product (range (0, (k_i - 1) * cell_size, cell_size),
                                   range (0, (k_j - 1) * cell_size, cell_size)):
        bin_cells[index] = ang[i : i + cell_size, j : j + cell_size, 0]
        mag_cells[index] = mag[i : i + cell_size, j : j + cell_size]
        index = index + 1

    # histograms calculating
    bins = [i * (180 // bin_n) for i in range (bin_n + 1)]
    hists = [np.histogram (b.ravel(), bins, weights =  m.ravel(), density = False)[0] for b, m in zip (bin_cells, mag_cells)]

    hist = [np.hstack([hists[0], hists[0], hists[0], hists[0]])] * (k_i - 2) * (k_j - 2)
    # block normalization
    index = 0
    for i, j in itertools.product (range (0, k_i - 2, 1), range (0, k_j - 2, 1)):
        block = np.hstack([hists[i * (k_i - 1) + j],
                           hists[i * (k_i - 1) + j + 1],
                           hists[(i + 1) * (k_i - 1) + j],
                           hists[(i + 1) * (k_i - 1) + j + 1]])
        norma = np.linalg.norm (block)
        if (norma > 0):
            block /= norma
        hist[index] = block
        index = index + 1

    # casting to one vector
    return np.hstack (hist)