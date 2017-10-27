import cv2
from laneDetection import LaneTracker
from tools import pipeline
import matplotlib.pyplot as plt
from imageTransformation import calibrate_camera
from moviepy.editor import VideoFileClip

### Calibrate the camera

ret, mtx, dist = calibrate_camera()

#load a sample image
img = cv2.imread('./test_images/test6.jpg')
imgRGB = img[:,:,::-1]
#plt.imshow(imgRGB)
#plt.show()
laneTracker = LaneTracker(mtx, dist)
augmented_image = laneTracker.process_image(img)
print(augmented_image.shape)
plt.imshow(augmented_image)
plt.show()

clip1 = VideoFileClip("./project_video.mp4")
res_video = clip1.fl_image(laneTracker.process_image) #NOTE: this function expects color images!!
res_video.write_videofile("./project_video_res.mp4", audio=False)