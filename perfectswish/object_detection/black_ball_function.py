import cv2
import numpy as np
from typing import List, Tuple
Image = np.ndarray
import matplotlib.pyplot as plt


def hough_circles(image, min_radius, max_radius, min_dist, param1, param2):
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1.5, min_dist, param1=param1, param2=param2,
                               minRadius=min_radius, maxRadius=max_radius)
    return circles

def detect_black_ball(image: Image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # value = 255-value
    h, s, v = cv2.split(image_hsv)
    v = 255 - v
    image_hsv = cv2.merge((h, s, v))
    image_rgb = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
    image = cv2.bitwise_or(image_hsv, image_rgb)
    plt.imshow(image)
    plt.show()



image = cv2.imread("black_ball_img.jpg")
detect_black_ball(image)
