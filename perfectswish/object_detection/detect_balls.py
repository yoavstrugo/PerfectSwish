import cv2
import numpy as np
import matplotlib.pyplot as plt

from perfectswish.object_detection.detect_cuestick import CuestickDetector
from perfectswish.utils.frame_buffer import FrameBuffer
from perfectswish.utils.utils import Colors, show_im
from perfectswish.image_transformation.image_processing import transform_board
def draw_circles(image, circles):
    if circles is not None:
        for i in circles:
            # print(i)
            cv2.circle(image, (int(i[0]), int(i[1])), 30, (0, 255, 0), 3)
    return image

def remove_fiducials(image, back_fiducial_id, front_fiducial_id):
    fiducial_detector = CuestickDetector(back_fiducial_id=back_fiducial_id, front_fiducial_id=front_fiducial_id)
    cuestick = fiducial_detector.detect_cuestick(image)
    if cuestick is not None:
        stickend, back_fiducial_center, front_fiducial_center = cuestick
        image_copy = image.copy()
        cv2.circle(image_copy, tuple(np.int32(back_fiducial_center)), 70, (150, 200, 100), -1)
        cv2.circle(image_copy, tuple(np.int32(front_fiducial_center)), 70, (150, 200, 100), -1)
        return image_copy
    return image


import copy
def remove_green(_image, new_color = Colors.BLACK):
    image = copy.deepcopy(_image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 0, 0])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    image[mask == 255] = new_color
    return image

# for every white pixel, set the pixel to the average of the surrounding pixels
# TODO this is a very naive implementation, it can be improved
def remove_white(image):
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            if np.linalg.norm(image[i, j]) >= 200:
                image[i, j] = image[i - 1:i + 2, j - 1:j + 2].mean(axis=(0, 1))
    return image

def bilateral_filter(image):
    return cv2.bilateralFilter(image, 9, 1, 200)

def canny_edge_detection(image):
    return cv2.Canny(image, 300, 400)

def hough_circles(image, min_radius, max_radius, min_dist, param1, param2):
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1.5, min_dist, param1=param1, param2=param2,
                               minRadius=min_radius, maxRadius=max_radius)
    return circles


def find_balls(image, new_color=Colors.GREEN, return_intermediates=False):

    # remove the fiducials
    image = remove_fiducials(image, 3, 4)
    # Remove green from the image
    no_green_ = remove_green(image, new_color=new_color)
    # Remove white pixels
    # no_white = remove_white(no_green)
    # Apply bilateral filter
    bilat_ = bilateral_filter(no_green_)
    # Apply canny edge detection
    canny_ = canny_edge_detection(bilat_)
    # Apply hough circles
    circles = hough_circles(canny_, 15, 30, 20, 50, 30)
    # return only the xy coordinates of the circles:
    if circles is not None:
        circles = circles[0, :][:, :2]
        circles = np.unique(circles, axis=0)

    if return_intermediates:
        # Return the intermediate results and the final circles
        data = draw_intermediate_images(bilat_, canny_, image, no_green_)
        return data, circles
    else:
        return circles


def draw_intermediate_images(bilat_, canny_, image, no_green_):
    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    ax[0].imshow(cv2.cvtColor(no_green_, cv2.COLOR_BGR2RGB))
    ax[0].axis('off')
    ax[0].set_title("No green")
    ax[1].imshow(cv2.cvtColor(bilat_, cv2.COLOR_BGR2RGB))
    ax[1].axis('off')
    ax[1].set_title("Bilateral filter")
    ax[2].imshow(canny_, cmap='gray')
    ax[2].axis('off')
    ax[2].set_title("Canny edge detection")
    balls = find_balls(image, new_color=[0, 255, 0])
    ax[3].imshow(draw_circles(image, balls), cmap='gray')
    ax[3].axis('off')
    ax[3].set_title("Hough circles")
    # make the plot a numpy image
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data

if __name__ == '__main__':

    from perfectswish.utils import webcam

    cap = webcam.initialize_webcam(1)

    buffer = FrameBuffer(5)

    while True:
        frame = webcam.get_webcam_image(cap)
        rect = [72, 972, 105, 127, 1800, 152, 1817, 1005]
        cropped_image = transform_board(frame, rect)
        # transpose the image

        buffer.add_frame(cropped_image)
        average_frame = buffer.get_average_frame()
        if average_frame is not None:

            intermediate, circles = find_balls(average_frame, new_color=[0, 255, 0], return_intermediates=True)
            if circles is not None:
                average_frame = draw_circles(average_frame, circles)
            cv2.imshow("average_frame", intermediate)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break