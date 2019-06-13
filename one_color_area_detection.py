# Python3

import cv2 as cv
import numpy as np

import image_info
import image_viewer

class rectangle_controller:
    i1 = (0, 0)
    i2 = (0, 0)

    def __init__(self, image):
        self.img = image
        self.drawing_right_now = False  # true if mouse is pressed
        self.drew = False
        self.ix, self.iy = -1, -1

        self.i1 = (self.ix, self.iy)
        self.i2 = (self.ix, self.iy)

    def mouse_action (self, event, x, y, flags, param):

        if event == cv.EVENT_LBUTTONDOWN:
            self.drawing_right_now = True
            self.ix, self.iy = x, y
            self.i1 = (x, y)

        elif event == cv.EVENT_MOUSEMOVE:
            if self.drawing_right_now == True:
                cv.rectangle (self.img, (self.ix, self.iy), (x, y), (0, 255, 0), -1)

        elif event == cv.EVENT_LBUTTONUP:
            self.drawing_right_now = False
            cv.rectangle (self.img, (self.ix, self.iy), (x, y), (0, 255, 0), -1)
            self.i2 = (x, y)
            self.drew = True

    ######## First common part
my_image = './pics/model.jpg'


    ######## Processing
if __name__ == '__main__':
    img = cv.imread (my_image)
    rect_ctrl = rectangle_controller (img)
    image_viewer.view_image_on_window (img, rect_ctrl)

    img = cv.imread (my_image)
    hsv_img = cv.cvtColor (img, cv.COLOR_BGR2HSV)

    conf_ints = image_info.get_confidence_intervals (hsv_img, rect_ctrl.i1, rect_ctrl.i2)
    low = np.array([conf_ints[k][0] for k in (0, 1, 2)])
    high = np.array ([conf_ints[k][1] for k in (0, 1, 2)])
    #print (conf_ints)
    #low, high = image_info.find_avg_low_high (hsv_img, rect_ctrl.i1, rect_ctrl.i2)

    print (low)
    print (high)

    mask = cv.inRange (hsv_img, low, high)
    hsv_img[mask > 0] = ([75, 255, 200])

    image_viewer.view_image_on_window (hsv_img)

    rgb_img = cv.cvtColor (hsv_img, cv.COLOR_HSV2RGB)
    gray_img = cv.cvtColor (rgb_img, cv.COLOR_RGB2GRAY)
    image_viewer.view_image_on_window (gray_img)

    ret, threshold = cv.threshold (gray_img, 90, 255, 0)
    image_viewer.view_image_on_window(threshold)

    contours, hierarchy = cv.findContours (threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours (img, contours, -1, (0, 0, 255), 3)
    image_viewer.view_image_on_window(img)


