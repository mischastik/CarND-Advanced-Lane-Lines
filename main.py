import cv2
import os
import numpy as np
from os.path import isfile, join
from tools import pipeline
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

### PART 1: Calibrate the camera

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

    objpoints = np.zeros((9*6, 3), np.float32)
    objpoints[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
    for img_filename in img_paths:
        #load them and detect the corners
        img_path = join(calibration_dir, img_filename)
        if not isfile(img_path):
            continue
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        if (ret):
            img_corners.append(corners)
            obj_corners.append(objpoints)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_corners, img_corners, gray.shape[::-1], None, None)

    pickle.dump([ret, mtx, dist, rvecs, tvecs], open("calibration.p", "wb"))

#load a smaple image
img = cv2.imread('./test_images/test2.jpg')
imgRGB = img[:,:,::-1]
#plt.imshow(imgRGB)
#plt.show()

### PART 2: Amplify lines
combined = pipeline(img, s_thresh=(60, 255), sm_thresh=(50, 130), sd_thresh=(1.15, 1.15))
combined = np.logical_or(np.squeeze(combined[:,:,1]), np.squeeze(combined[:,:,2]))
combined = combined.astype(np.float32)
#print(combined.shape)
#plt.imshow(combined)
#plt.show()


### PART 3: Undistort and rectify images:
#LL: 286 664, LR: 286+751=1037,664, UL: 586,455, UR: 586+113=699,455

#undistort it
undistorted = cv2.undistort(combined, mtx, dist, None, mtx)
#plt.imshow(undistorted)
#plt.show()
#rectify street surface
border_size = 400
dst = np.float32([[border_size, border_size], [1000 + border_size, border_size], [1000 + border_size, 1000 + 2* border_size], [border_size, 1000 + 2 * border_size]])
src = np.float32([[586, 455], [699, 455], [1037, 664], [286, 664]])
M = cv2.getPerspectiveTransform(src, dst)
warped = cv2.warpPerspective(undistorted, M, (1000 + 2*border_size, 1000 + 2 * border_size))
plt.imshow(warped)
plt.show()
