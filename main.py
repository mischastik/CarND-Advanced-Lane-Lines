import cv2
import numpy as np

from tools import pipeline
import matplotlib.pyplot as plt

from laneDetection import findLaneLines, trackLaneLines, measure_curvature, measure_offset, computeMetricPolyCoeffs
from imageTransformation import calibrate_camera

### PART 1: Calibrate the camera

ret, mtx, dist = calibrate_camera()

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
dst = np.float32([[border_size, 2 * border_size], [1000 + border_size, 2* border_size], [1000 + border_size, 1000 + 2*border_size], [border_size, 1000+2*border_size]])
src = np.float32([[583, 460], [704, 460], [1049, 680], [275, 680]])
# height is two dashes and two gaps = 18m (assuming gaps are approx. 9m and line segments are approx. 3m)
# width is from line centers = 3.7m
ym_per_pix = 24.384 / 1000
xm_per_pix = 3.7 / 1000

M = cv2.getPerspectiveTransform(src, dst)
warped = cv2.warpPerspective(undistorted, M, (1000 + 2*border_size, 1000 + 2*border_size))
plt.imshow(warped)
#plt.show()
[left_fit, right_fit, out_img] = findLaneLines(warped)

left_fit_cr = computeMetricPolyCoeffs(left_fit, warped.shape[0], xm_per_pix, ym_per_pix)
right_fit_cr = computeMetricPolyCoeffs(right_fit, warped.shape[0], xm_per_pix, ym_per_pix)

# Generate x and y values for plotting
ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, out_img.shape[1])
plt.ylim(out_img.shape[0], 0)
#plt.show()

[left_fit, right_fit] = trackLaneLines(warped, left_fit, right_fit)

curvature_left = measure_curvature(left_fit_cr, y=(warped.shape[0] - 1)*ym_per_pix)
curvature_right = measure_curvature(right_fit_cr, y=(warped.shape[0] - 1)*ym_per_pix)
offset = measure_offset(left_fit_cr, right_fit_cr, y=(warped.shape[0] - 1)*ym_per_pix, width=warped.shape[1]*xm_per_pix)
print("cL: {0}; cR: {1}, offset: {2}".format(curvature_left, curvature_right, offset))
# Generate x and y values for plotting
ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
