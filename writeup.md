**Advanced Lane Finding Project**

[//]: # (Image References)

[image1]: ./writeup_materials/calibration3.jpg "Original"
[image11]: ./writeup_materials/calibration3_undistorted.jpg "Undistorted"

[image2]: ./writeup_materials/01_input.png "Sample input"

[image31]: ./writeup_materials/02tresholded.png "Thresholded output"
[image32]: ./writeup_materials/03thresholded_after_AND.png "Thresholded final"

[image4]: ./writeup_materials/04undistorted.png "Undistortion Example"

[image5]: ./writeup_materials/05warped.png "Warped image"

[image61]: ./writeup_materials/07polynomial_fit.png "Sliding window search"
[image62]: ./writeup_materials/09_tracking.png "Tracking result"

[image7]: ./writeup_materials/10_final_result.png "Final result"

[video1]: ./project_video.mp4 "Video"

### Camera Calibration

Camera images are usually distorted because of imperfect projections by the camera lens. With a properly calibrated lens model, these distortions can be reverted algorithmically. This step is mandatory for lane finding because we need to ensure that line in the world space are projected onto lines in the image space to calculate curvature correctly.
First we need to detemine the parameters of the lens model in order to undistort the camera image. This can be done by taking multiple images of a regular checkerboard. For good results, it is absolutely mandatory that the surface of the checkerboard is absolutely flat and that the corner distances are homogeneous up to a few micrometers.
The camera calibration is done in method the 'calibrate_camera' in 'imageTransformation.py'.
For calibration we need to object points and the according coordinates of the projected corners in image space. 
The object points are the 3-D corner points in "checkerboard space" with the z-coodinate being 0. In this case it's simply a meshgrid of 9x6 elements with an increment of one since the true metric corner distance is unknown and not required for this calibration. 
The projected coordinates inner checkerboard corners are detected with OpenCV's 'findChessboardCorners'.
The calibration is done with 'calibrateCamera'. It takes to object points and the projected pixel positions in all the images. A optimization algorithm then computes the extrinsic (rigid-body-transformation) parameters of each checkerboard and the intrinsic (lens model) parameters of the camera. The checkerboard extrinsics won't be correct in this case because we didn't use the correct metric distances for the object points list. However, we don't need this for the lane finding anyway. 
Once the model parameters are determined, the calibration is then cached so that it's not computed every time. Please keep in mind that the cached file 'calibration.p' needs to be deleted if a new calibration should be made.

OpenCV's 'undistort' method can apply these model parameters to compute an undistorted representation of the image data. This image has zero tangential and radial distortion parameters and basically satisfies a pinhole camera model. This means, lines in world space are actually projected onto lines in image space. The following illustration  shows the effect of undistortion on one of the calibration images:

![Original checkerboard image][image1]
![Undistorted checkerboard image][image11]

### Lane Detection

#### Overview
The lane detection algorithm consists of several stages:
* Lane marker enhancement: Thresholds to edge and color channels are applied to determine candidate pixels for lane markers.  
* Undistortion: The image is rectified so this lines are projected to lines.
* Warping: An orthogonal projection of the street surface in front of the car is computed.
* Line detection: 
** Initialization: A 2-D to 1-D projection is computed in a window of full width and part of the height along the y-axis of the warped image. The two local maxima in each window are calculated to find the approximate line position.
** Polynomial fit: A polynomial is fit through each of the approximate line positons. The polynomials should represent the left and right lane markers.
** Quality assessment: The detected lane markers are evaluated for geometric consistency. 
* Line tracking: If a lane was consistently detected in the previous frame, this detection is taken as a basis for tracking.
** Candidate extraction: Candidate pixels are selected in the proximity of the valid previously detected lane markers.
** Polynomial fit: A polynomial is fit through each of the candidate regions.

#### Sample Input
The follwing image was used as an input for all illustrations of the stages of the algorithm: 

![Input][image2]


#### Lane Marker Enhancement:

This part can be found in method 'thresholding' in 'tools.py'.

For lane marker enhancement I first tried thresholding edge magnitude, edge direction and the L channel from the LSV color space.
Edge direction turned out to be too unspecific and was dropped. Edge magnitude and L channel thresholding worked very well on most of the test images and on 'project_video.mp4'. It didn't work well on 'challenge_video.mp4' though. The R channel from RGB seems to be a more robust candidate. The edge magnitude still was rather specific but thinned out the lines a little too much even at are carefully chosen thresholds, so a morphologic dilation was applied to the thresholded result. The thresholds on the red channel and the gradient magnitude ware chosen so that each candidate pixel should be contained in both thresholded inputs (logical AND). This turned out to be more robust than choosing stricter thresholds and require only one image to contain the candidate (logical OR). 
This leads to the following candidates (green channel is edge thresholded, blue channel is intensity thresholded red channel):

![Thresholded candidates][image31]

After applying an AND operation we get this final thresholded image:

![Thresholded final output][image32]


#### Undistortion:

This image is then undistorted with the previously calculated lens model parameters:

![Result after undistortion][image4]

#### Warping:
For computing an orthogonal projection of the street surface in front of the car we use OpenCVs 'getPerspectiveTransform' and 'warpPerspective'.
The output image should be 1000 wide for the lane and a configurable border of border_size to the left and right (we chose 400) in our case. The choice of the border size depends on the maximum curvature expected and the distance we want to look ahead. The later should be mapped to 1000 pixels in the warped image. We tried three different look ahead distances based on the distance of the dashed on the road:
* Short: one dash and one gap (approx. 12m)
* Medium: two dashes and one gap (approx. 15m)
* Far: two dashes and two gaps (approx. 24m)

For the test data the medium distance seemed to be appropriate which leads to this mapping of source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 559, 475      | 400, 0 | 
| 729, 475      | 1400, 0      |
| 1053, 681     | 1400, 1000      |
| 174, 681      | 400, 1000        |

The result image should be 1400x1000 pixels large and have a vertical resultion of 15 m per 1000 pixels and a horizontal resolution of 3.7 m per 1000 pixels. 
In video 'challenge_video.mp4' the alignment of the camera seemed to be sightly different so we added a correction term that allows stretching or shrinking the far end to match slightly different alignments.

Applying this transformation gives us the following image:

![Warped image][image5]

#### Line Detection

We initialize the line detection by computing 2-D to 1-D projection in a sliding window along the y-axis of the warped image.
The window spans the full width and part of the height. The two local maxima in each window are calculated to find the approximate line position. All points nearby the left and right local maxima within the window are chosen as candidates. 
Then a polynomial is fit through the candidates. The polynomials should represent the left and right lane markers.

The sliding window (green), the left and right candidate pixels (blue and red) and the polynomial fit result (yellow) can be seen in this image: 

![Sliding window result][image61]

#### Line Tracking
Tracking works essentially the same way but we skip the sliding window candidate search and determine the candidates by selecting all pixels nearby the previously detected polynomial

![Tracking result][image62]

#### Detection Quality Check
In method 'evaluate_detection_quality' the detection quality is assured by computing the metrically correct polynomial using the resolution values for the warped image.
Then the curvature of left and right lane marker is computed and the offset to the center:
* The minimum allowed curvature for both markers is set to 250 m.
* The maximum allowed offset to the center is 2 m.
* The minimum allowed distance between the lines is 2.7 m, the maximum is 4.7 m (the lane width is assumed to be 3.7 m).

If one of these criteria is not met, the detection is discarded.
If the detection is considered valid, the estimate of the lane is updated with an infinite-impulse response filter with an update rate of 0.5.

#### Backprojection
The final result is then backprojected onto the original image and a polygon is drawn onto the detected lane:

![Final result][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_re.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
