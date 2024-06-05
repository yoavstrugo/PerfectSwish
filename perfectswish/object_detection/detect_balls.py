import cv2
import numpy as np
import matplotlib.pyplot as plt
from perfectswish.api.frame_buffer import FrameBuffer
from perfectswish.api.utils import Colors, show_im

def transform_board(image, rect):
    # Get the coordinates of the corners of the board
    x1, y1, x2, y2, x3, y3, x4, y4 = rect

    # Set the target size for the new image
    RESOLUTION_FACTOR = 8
    target_width = 112 * RESOLUTION_FACTOR
    target_height = 224 * RESOLUTION_FACTOR

    # Define the new coordinates of the corners in the new image
    new_rect = np.array([[0, 0], [target_width, 0], [target_width, target_height], [0, target_height]],
                        dtype=np.float32)

    # Calculate the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32), new_rect)

    # Apply the perspective transformation to the original image
    transformed_image = cv2.warpPerspective(image, matrix, (target_width, target_height))

    return transformed_image

def draw_circles(image, circles):
    if circles is not None:
        for i in circles:
            print(i)
            cv2.circle(image, (int(i[0]), int(i[1])), 20, (0, 255, 0), 3)
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

    from perfectswish.api import webcam

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