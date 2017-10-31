**Advanced Lane Finding Project**

[//]: # (Image References)

[image1]: ./writeup_materials/calibration3.jpg "Original"
[image11]: ./writeup_materials/calibration3_undistorted.jpg "Undistorted"

[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
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

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
