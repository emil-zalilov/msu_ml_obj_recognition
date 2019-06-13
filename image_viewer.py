# Python3

import cv2 as cv

def view_image_on_window (image, action_controller = None) :
    window_name = 'Display'
    cv.namedWindow (window_name, cv.WINDOW_NORMAL)
    if action_controller:
        cv.setMouseCallback (window_name, action_controller.mouse_action)

    while (1):
        cv.imshow (window_name, image)
        k = cv.waitKey (20) & 0xFF
        flag = False
        if action_controller is not None:
            flag = action_controller.drew
        if k == 27 or flag:
            break
    cv.destroyAllWindows ()
