from os.path import isfile, join
import os
import pickle
import numpy as np
from pathlib import Path
import cv2

def calibrate_camera():
    my_file = Path("calibration.p")
    # if we already have a trained model, load it and continue training
    # important: Remember to delete model.h5 if the model architecture is changes, otherwise changes won't take effect
    if my_file.exists():
        [ret, mtx, dist, rvecs, tvecs] = pickle.load(open("calibration.p", "rb"))
    else:
        calibration_images = []
        img_corners = []
        obj_corners = []

        calibration_dir = './camera_cal'
        # find all calibration images
        img_paths = os.listdir(calibration_dir)

        objpoints = np.zeros((9 * 6, 3), np.float32)
        objpoints[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        for img_filename in img_paths:
            # load them and detect the corners
            img_path = join(calibration_dir, img_filename)
            if not isfile(img_path):
                continue
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
            if (ret):
                img_corners.append(corners)
                obj_corners.append(objpoints)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_corners, img_corners, gray.shape[::-1], None, None)

        pickle.dump([ret, mtx, dist, rvecs, tvecs], open("calibration.p", "wb"))

    return ret, mtx, dist