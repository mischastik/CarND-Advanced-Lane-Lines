import cv2
from laneDetection import LaneTracker
from tools import thresholding
import matplotlib.pyplot as plt
from imageTransformation import calibrate_camera
from moviepy.editor import VideoFileClip

### Calibrate the camera

ret, mtx, dist = calibrate_camera()

### SAMPLE UNDISTORTION
#img = cv2.imread('./camera_cal/calibration3.jpg')
#undistorted_cb = cv2.undistort(img, mtx, dist, None, mtx)
#cv2.imwrite('./writeup_materials/calibration3_undistorted.jpg',undistorted_cb)

laneTracker = LaneTracker(mtx, dist)

### SAMPLE IMAGE PROCESSING
#load a sample image
#img = cv2.imread('./challenge.png')
#imgRGB = img
#img = cv2.imread('./test_images/test6.jpg')
#imgRGB = img[:,:,::-1]
#plt.imshow(imgRGB)
#plt.show()
#augmented_image = laneTracker.process_image(imgRGB)
#plt.imshow(augmented_image)
#plt.show()

### VIDEO PROCESSING
clip1 = VideoFileClip("./project_video.mp4")
res_video = clip1.fl_image(laneTracker.process_image) #NOTE: this function expects color images!!
res_video.write_videofile("./project_video_res.mp4", audio=False)

laneTracker.alignment_correction = 40
clip1 = VideoFileClip("./challenge_video.mp4")
res_video = clip1.fl_image(laneTracker.process_image) #NOTE: this function expects color images!!
res_video.write_videofile("./challenge_video_res.mp4", audio=False)