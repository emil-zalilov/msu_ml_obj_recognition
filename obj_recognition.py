import numpy as np
import cv2 as cv
import os
import time

import hog

### settings of image
ROWS = 256
COLS = 256

is_intensity = False
is_knn = False

training_set_dir = "./dataset/Training_set"
test_set_dir = "./dataset/test"

classes = os.listdir (training_set_dir)
classes.remove ('.DS_Store')
print (classes)
start_time = time.time ()
print ('Start time: %s' % time.ctime (start_time))

######### Construct training data

def append_data (X, Y, img, class_name):
    if (is_intensity):
        append_data_intensity (X, Y, img, class_name)
    else:
        append_data_hog (X, Y, img, class_name)

### knn
def append_data_intensity (X, Y, img, class_name):
    gray_img = cv.cvtColor (img, cv.COLOR_BGR2GRAY)

    X.append (np.array (gray_img[: , :]).reshape (ROWS * COLS).astype(np.float32))
    Y.append (np.array ([classes.index (class_name)]).astype (np.int32))

### svm
def append_data_hog (X, Y, img, class_name):
    hogdata = hog.hog (np.float32 (img) / 255.0)

    X.append (np.float32 (hogdata))
    Y.append (np.array ([classes.index (class_name)]).astype (np.int32))


training_X = []
training_Y = []
test_X = []
test_Y = []

def classes_loop (dir, X, Y) :
    for class_name in classes:
        files = os.listdir (dir + '/' + class_name)
        for file in files:
            img_t = cv.imread (dir + '/' + class_name + '/' + file)
            append_data (X, Y, cv.resize (img_t, (ROWS, COLS), interpolation = cv.INTER_CUBIC), class_name)

train_data = np.load ('hog_train_data.npz')
test_data = np.load ('hog_test_data.npz')

if not len (train_data.files) == 0:
    training_X = train_data['training_X']
    training_Y = train_data['training_Y']

    train_data.close ()
else:
    classes_loop (training_set_dir, training_X, training_Y)
    training_X = np.array (training_X)
    training_Y = np.array (training_Y)

    if (is_intensity):
        np.savez ('intensity_train_data.npz', training_X = training_X, training_Y = training_Y)
    else:
        np.savez ('hog_train_data.npz', training_X = training_X, training_Y = training_Y)


if not len (test_data.files) == 0:
    test_X = test_data['test_X']
    test_Y = test_data['test_Y']
else:
    classes_loop (test_set_dir, test_X, test_Y)
    test_X = np.array (test_X)
    test_Y = np.array (test_Y)

    if (is_intensity):
        np.savez ('intensity_test_data.npz', test_X = test_X, test_Y = test_Y)
    else:
        np.savez ('hog_test_data.npz', test_X = test_X, test_Y = test_Y)

############ Processing

if (is_knn):
    knn = cv.ml.KNearest_create ()
    knn.train (training_X, cv.ml.ROW_SAMPLE, training_Y)
    k = 16
    ret,result,neighbours,dist = knn.findNearest (test_X, k)
else:
    svm = cv.ml.SVM_create ()
    svm.setKernel (cv.ml.SVM_POLY)
    svm.setType (cv.ml.SVM_C_SVC)
    degree = 2.347
    coef0 = 178
    C = 2.67
    gamma = 150.383
    svm.setDegree (degree)
    svm.setCoef0 (coef0)
    svm.setC (C)
    svm.setGamma (gamma)

    print ("Params: ", degree, coef0, C, gamma)

    svm.train (training_X, cv.ml.ROW_SAMPLE, training_Y)
    result = svm.predict (test_X)[1]

matches = result == test_Y
correct = np.count_nonzero (matches)
accuracy = correct*100.0 / result.size
print (accuracy)
total_time = time.time () - start_time
print ('Total time: %s' % total_time)

