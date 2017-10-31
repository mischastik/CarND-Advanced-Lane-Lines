import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from tools import thresholding
from Lane import Line

class LaneTracker:
    def __init__(self, mtx, dist, alignment_correction = 0):
        self.xm_per_pix = 1
        self.ym_per_pix = 1
        self.mtx = mtx
        self.dist = dist
        self.line_left = Line()
        self.line_right = Line()
        self.alignment_correction = alignment_correction
        self.frame_count = 0


    # this is the key method which does all the lane finding and tracking, the rest should be private
    def process_image(self, img):
        ### Extract line candidate pixels
        combined = thresholding(img, s_thresh=(150, 255), sm_thresh=(20, 255), sd_thresh=(1.15, 1.15))

        #plt.imshow(combined)
        #plt.show()

        combined = np.logical_and(np.squeeze(combined[:, :, 1]), np.squeeze(combined[:, :, 2]))
        combined = combined.astype(np.float32)

        # print(combined.shape)
        #plt.imshow(combined, cmap='gray')
        #plt.show()


        ### Undistort and rectify images:
        # Coordinates of long frustum intersection: LL: 286 664, LR: 286+751=1037,664, UL: 586,455, UR: 586+113=699,455
        # Coordinates of medium frustum intersection: LL: 274 681, LR: 1053 681, UL: 559 475, UR: 729 475
        # Coordinates of short frustum intersection: LL: 274 681, LR: 1053 681, UL: 540 490, UR: 751 490
        # undistort image
        undistorted = cv2.undistort(combined, self.mtx, self.dist, None, self.mtx)
        undistorted_orig = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        #cv2.cvtColor(undistorted_orig, undistorted_orig, cv2.COLOR_BGR2RGB)
        #plt.imshow(undistorted, cmap='gray')
        #plt.show()
        # rectify street surface
        border_size = 400

        #dst = np.float32([[border_size, 2 * border_size], [1000 + border_size, 2 * border_size],
        #                  [1000 + border_size, 1000 + 2 * border_size], [border_size, 1000 + 2 * border_size]])
        dst = np.float32([[border_size, 0], [1000 + border_size, 0],
                          [1000 + border_size, 1000], [border_size, 1000]])
        #src = np.float32([[583, 460], [704, 460], [1049, 680], [275, 680]])
        src = np.float32([[559+self.alignment_correction, 475], [729-self.alignment_correction, 475], [1053, 681], [274, 681]])
        #src = np.float32([[540, 490], [751, 490], [1053, 681], [274, 681]])
        # Long: height is two dashes and two gaps = 24m (assuming gaps are approx. 9m and line segments are approx. 3m)
        # Medium: height is two dashes and one gap = 15m (assuming gaps are approx. 9m and line segments are approx. 3m)
        # Short: height is one dash and one gap = 12m (assuming gaps are approx. 9m and line segments are approx. 3m)
        # width is from line centers = 3.7m
        #self.ym_per_pix = 24.384 / 1000
        self.ym_per_pix = 15.0 / 1000.0
        self.xm_per_pix = 3.7 / 1000.0

        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        #warped = cv2.warpPerspective(undistorted, M, (1000 + 2 * border_size, 1000 + 2 * border_size))
        warped = cv2.warpPerspective(undistorted, M, (1000 + 2 * border_size, 1000))
        #if self.frame_count > 323:
        #plt.imshow(warped)
        #plt.show()
        if self.line_left.detected:
            [left_fit, right_fit] = self.trackLaneLines(warped, self.line_left.current_fit, self.line_right.current_fit)
        else:
            [left_fit, right_fit, out_img] = self.findLaneLines(warped)
        #plt.imshow(out_img)
        #plt.show()
        if left_fit is None or right_fit is None:
            self.line_left.detected = False
            self.line_right.detected = False
            return img

        left_fit_cr = self.computeMetricPolyCoeffs(left_fit, warped.shape[0])
        right_fit_cr = self.computeMetricPolyCoeffs(right_fit, warped.shape[0])

        if self.evaluate_detection_quality(left_fit_cr, right_fit_cr, warped.shape[0], warped.shape[1]):
            #detection is ok:
            self.line_left.add_valid_fit(left_fit, left_fit_cr)
            self.line_right.add_valid_fit(right_fit, right_fit_cr)
        else:
            self.line_left.detected = False
            self.line_right.detected = False
            # use previous results instead:
            if len(self.line_left.recent_fits) > 0:
                left_fit = self.line_left.current_fit
                right_fit = self.line_right.current_fit
                left_fit_cr = self.line_left.current_fit_cr
                right_fit_cr = self.line_right.current_fit_cr

        if self.line_left.iir_average_fit is not None:
            left_fit = self.line_left.iir_average_fit
        if self.line_right.iir_average_fit is not None:
            right_fit = self.line_right.iir_average_fit
        # [left_fit, right_fit] = trackLaneLines(warped, left_fit, right_fit)

        #print("cL: {0}; cR: {1}, offset: {2}".format(curvature_left, curvature_right, offset))
        # Generate x and y values for plotting
        ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        #fig = Figure()
        #canvas = FigureCanvas(fig)
        #ax = fig.gca()
        #ax.axis('off')

        #plt.imshow(out_img)
        #plt.plot(left_fitx, ploty, color='yellow')
        #plt.plot(right_fitx, ploty, color='yellow')
        #plt.xlim(0, out_img.shape[1])
        #plt.ylim(out_img.shape[0], 0)
        #canvas.draw()  # draw the canvas, cache the renderer

        #plt_width, plt_height = fig.get_size_inches() * fig.get_dpi()

        #augmented_image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
        #augmented_image = augmented_image.reshape(plt_height, plt_width, 3)
        #canvas.draw()
        #buf = canvas.tostring_rgb()
        #ncols, nrows = canvas.get_width_height()
        #return np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        if self.line_left.detected:
            cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        else:
            cv2.fillPoly(color_warp, np.int_([pts]), (32, 255, 0))


        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        #undistorted_orig = undistorted_orig[:, :, ::-1]
        newwarp = cv2.warpPerspective(color_warp, Minv, (undistorted_orig.shape[1], undistorted_orig.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(undistorted_orig, 1, newwarp, 0.3, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        curvature_left = LaneTracker.measure_curvature(left_fit_cr, y=(warped.shape[0] - 1) * self.ym_per_pix)
        curvature_right = LaneTracker.measure_curvature(right_fit_cr, y=(warped.shape[0] - 1) * self.ym_per_pix)
        offset = self.measure_offset(left_fit_cr, right_fit_cr, y=(warped.shape[0] - 1) * self.ym_per_pix,
                                width=warped.shape[1] * self.xm_per_pix)
        dst = right_fit_cr[2] - left_fit_cr[2]
        cv2.putText(result, 'cL: {0:.2f}'.format(curvature_left), (10, 70), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(result, 'cR: {0:.2f}'.format(curvature_right), (10, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(result, 'offs: {0:.2f}'.format(offset), (10, 130), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(result, 'dst: {0:.2f} '.format(dst), (10, 160), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        #plt.imshow(result)
        #plt.show()
        self.frame_count += 1
        return result

    def findLaneLines(self, binary_warped):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        # plt.plot(histogram)
        # plt.show()
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        out_img = out_img.astype(np.uint8)
        #plt.imshow(out_img)
        #plt.show()
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 200
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                          (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                          (0, 255, 0), 2)
            # plt.imshow(out_img)
            # plt.show()
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        if lefty.size == 0 or righty.size == 0:
            return [None, None, out_img]
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        #cv2.imwrite('out_img.png', out_img)
        return [left_fit, right_fit, out_img]

    def trackLaneLines(self, binary_warped, left_fit, right_fit):
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        polyvals_left = left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2]
        left_lane_inds = ((nonzerox > (polyvals_left - margin)) & (nonzerox < (polyvals_left + margin)))
        polyvals_right = right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2]
        right_lane_inds = ((nonzerox > (polyvals_right - margin)) & (nonzerox < (polyvals_right + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit_new = np.polyfit(lefty, leftx, 2)
        right_fit_new = np.polyfit(righty, rightx, 2)

        ## Create an image to draw on and an image to show the selection window
        #ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        #left_fitx = left_fit_new[0] * ploty ** 2 + left_fit_new[1] * ploty + left_fit_new[2]
        #right_fitx = right_fit_new[0] * ploty ** 2 + right_fit_new[1] * ploty + right_fit_new[2]
        #out_img = np.dstack((binary_warped, binary_warped, binary_warped))  * 255
        #out_img = out_img.astype(np.ubyte)

        #window_img = np.zeros_like(out_img)
        #window_img = window_img.astype(np.ubyte)
        # Color in left and right line pixels
        #out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        #out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        #left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        #left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
        #                                                                ploty])))])
        #left_line_pts = np.hstack((left_line_window1, left_line_window2))
        #right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        #right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
        #                                                                 ploty])))])
        #right_line_pts = np.hstack((right_line_window1, right_line_window2))

        ## Draw the lane onto the warped blank image
        #cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        #cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))

        #result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        #plt.imshow(result)
        #plt.plot(left_fitx, ploty, color='yellow')
        #plt.plot(right_fitx, ploty, color='yellow')
        #plt.show()

        return [left_fit_new, right_fit_new]

    def computeMetricPolyCoeffs(self, poly_coeffs, height):
        ploty = np.linspace(0, height - 1, height)
        fitx = poly_coeffs[0] * ploty ** 2 + poly_coeffs[1] * ploty + poly_coeffs[2]

        return np.polyfit(ploty * self.ym_per_pix, fitx * self.xm_per_pix, 2)

    @staticmethod
    def measure_curvature(poly_coeffs, y):
        r_curve = (1 + (2 * poly_coeffs[0] * y + poly_coeffs[1]) ** 2) ** (2.0 / 3.0) / abs(2 * poly_coeffs[0])
        return r_curve

    @staticmethod
    def evaluate_poly(coeffs, y):
        return coeffs[0] * (y ** 2) + coeffs[1] * y + coeffs[2]

    def evaluate_detection_quality(self, left_fit_cr, right_fit_cr, height, width):
        curvature_left = LaneTracker.measure_curvature(left_fit_cr, y=(height - 1) * self.ym_per_pix)
        curvature_right = LaneTracker.measure_curvature(right_fit_cr, y=(height - 1) * self.ym_per_pix)
        offset = self.measure_offset(left_fit_cr, right_fit_cr, y=(height - 1) * self.ym_per_pix,
                                width=width * self.xm_per_pix)

        #if (abs(curvature_left - curvature_right) > 250):
        #    return False
        if curvature_right < 250 or curvature_right < 250:
            return False
        if (abs(offset) > 2):
            return False
        if (abs(right_fit_cr[2] - left_fit_cr[2] - 3.7)) > 1:
            return False

        return True

    def measure_offset(self, left_fit, right_fit, y, width):
        left_val = self.evaluate_poly(left_fit, y)
        right_val = self.evaluate_poly(right_fit, y)
        #print(left_val)
        #print(right_val)
        center = (left_val + right_val) / 2.0
        #print(center)
        return center - width / 2.0

