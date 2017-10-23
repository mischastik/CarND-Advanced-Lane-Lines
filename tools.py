import numpy as np
import cv2
import matplotlib.pyplot as plt

def pipeline(img, s_thresh=(0.3, 1.0), sm_thresh=(50, 130), sd_thresh=(0.7, 1.15)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Sobel
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=3)

    sobel_mag = np.sqrt(np.square(sobelx) + np.square(sobely))
    scaled_sobel = np.uint8(255 * sobel_mag / np.max(sobel_mag))

    # Threshold gradient magnitude
    sm_binary = np.zeros_like(scaled_sobel)
    sm_binary[(scaled_sobel >= sm_thresh[0]) & (scaled_sobel <= sm_thresh[1])] = 1
    sm_binary = cv2.morphologyEx(sm_binary, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    # Sobel again for gradient direction computation
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=7)
    sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=7)
    abs_sobelx = np.abs(sobelx)
    abs_sobely = np.abs(sobely)
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)

    # Create a binary mask where direction thresholds are met
    sd_binary = np.zeros_like(grad_dir)
    sd_binary[(grad_dir >= sd_thresh[0]) & (grad_dir <= sd_thresh[1])] = 1
    #remove some isolated noise pixels
    sd_binary = cv2.morphologyEx(sd_binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
    # Threshold saturation channel
    plt.imshow(s_channel)
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack channels
    color_binary = np.dstack((sd_binary, sm_binary, s_binary))

    return color_binary

