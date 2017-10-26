import cv2
from laneDetection import LaneTracker
from tools import pipeline
import matplotlib.pyplot as plt
from imageTransformation import calibrate_camera

### PART 1: Calibrate the camera

ret, mtx, dist = calibrate_camera()

#load a sample image
img = cv2.imread('./test_images/test2.jpg')
imgRGB = img[:,:,::-1]
#plt.imshow(imgRGB)
#plt.show()
laneTracker = LaneTracker(mtx, dist)
augmented_image = laneTracker.process_image(img)
print(augmented_image.shape)
plt.imshow(augmented_image)
plt.show()
