from time import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

from perfectswish.object_detection.detect_cuestick import CuestickDetector
from perfectswish.utils.frame_buffer import FrameBuffer
from perfectswish.utils.utils import Colors, show_im
from perfectswish.image_transformation.image_processing import transform_board

def timer_func(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        # print(f'Function {func.__name__!r} executed in {(t2 - t1):.4f}s')
        return result

    return wrap_func


black_ball_temp = cv2.imread(r"perfectswish/object_detection/balls_for_template_matching/black_ball_color_template.jpg")
white_ball_temp = cv2.imread(r"perfectswish/object_detection/balls_for_template_matching/white_ball_color_template.jpg")


def draw_circles(image, circles):
    if circles is not None:
        for i in circles:
            # print(i)
            cv2.circle(image, (int(i[0]), int(i[1])), 30, (0, 255, 0), 3)
    return image


def remove_fiducials(image, back_fiducial_id, front_fiducial_id):
    fiducial_detector = CuestickDetector(back_fiducial_id=back_fiducial_id, front_fiducial_id=front_fiducial_id)
    stickend, back_fiducial_center, front_fiducial_center = fiducial_detector.detect_cuestick(image)
    if stickend is not None and back_fiducial_center is not None and front_fiducial_center is not None:
        image_copy = image.copy()
        cv2.circle(image_copy, tuple(np.int32(back_fiducial_center)), 70, (150, 200, 100), -1)
        cv2.circle(image_copy, tuple(np.int32(front_fiducial_center)), 70, (150, 200, 100), -1)
        return image_copy
    return image


import copy


def remove_green(image, new_color=Colors.BLACK):
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

    if circles is not None:
        circles = circles[0, :][:, :2]

    # find the white ball and black ball seperately:
    # black_ball = find_black_ball(image, black_ball_temp)
    white_ball = find_white_ball(image, white_ball_temp)
    circles = np.vstack([circles, white_ball])

    circles = np.unique(circles, axis=0)

    if return_intermediates:
        # Return the intermediate results and the final circles
        data = draw_intermediate_images(bilat_, canny_, image, no_green_)
        return data, circles
    else:
        return circles


def draw_black_and_white_balls(image):
    black_ball = find_black_ball(image, black_ball_temp)
    white_ball = find_white_ball(image, white_ball_temp)
    cv2.circle(image, black_ball, 30, (0, 0, 0), 3)
    cv2.circle(image, white_ball, 30, (255, 255, 255), 3)
    return image


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
    ax[3].imshow(draw_black_and_white_balls(draw_circles(image, balls)), cmap='gray')
    ax[3].axis('off')
    ax[3].set_title("Hough circles")
    # make the plot a numpy image
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data


def gamma_correction(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def find_black_ball(image, black_ball_template=black_ball_temp):
    res = cv2.matchTemplate(image, black_ball_template, cv2.TM_CCOEFF_NORMED)
    # theres only one black ball in the image so we can just take the max value
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # correct the max location
    max_loc = (max_loc[0] + black_ball_template.shape[1] // 2, max_loc[1] + black_ball_template.shape[0] // 2)
    # draw a circle around the black ball
    cv2.circle(image, max_loc, 30, (0, 255, 0), 3)
    return max_loc


def find_white_ball(image, white_ball_template=white_ball_temp):
    res = cv2.matchTemplate(image, white_ball_template, cv2.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    max_loc = (max_loc[0] + white_ball_template.shape[1] // 2, max_loc[1] + white_ball_template.shape[0] // 2)
    cv2.circle(image, max_loc, 30, (0, 255, 0), 3)
    return max_loc


class BallBuffer:
    # the main idea is to add a gaussian to the buffer for every ball detected, like a probability space. then we can
    # take local maximas of the buffers sum to get the most probable location of the balls
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.shape = (1800, 900)
        self.balls_images_queue = np.zeros((1800, 900, buffer_size),
                                           dtype=np.uint8)  # the dimensions dont matter here if its larger than the image

    def __add_gaussian(self, balls_frame: np.array, ball, radius=31):
        x, y = ball
        blank_image = np.zeros_like(balls_frame)
        cv2.circle(blank_image, (x, y), 3, 255, -1)
        gaussian = cv2.GaussianBlur(blank_image, (radius, radius), sigmaX=0)

        # add the gaussian to the frame
        balls_frame = cv2.addWeighted(balls_frame, 1, gaussian, 1, 0)
        return balls_frame

    @timer_func
    def add_balls(self, balls):
        balls_frame = np.zeros(self.shape, dtype=np.uint8)
        for ball in balls:
            balls_frame = self.__add_gaussian(balls_frame, ball)

        # move the queue back by one, and add the new frame
        self.balls_images_queue[:, :, :-1] = self.balls_images_queue[:, :, 1:]
        self.balls_images_queue[:, :, -1] = balls_frame

    def __get_maximas(self, image):

        mask = np.where(image > 30, 1, 0)  # thresholded at 30
        image = image * mask

        # gaussian filter the image
        image = cv2.GaussianBlur(image, (9, 9), 0)

        # now find local maximas using a maximum filter
        maxima = cv2.dilate(image, np.ones((9, 9)))
        local_maximums = np.where((image == maxima) & (image > 0))
        if local_maximums[0].size == 0:
            return np.array([])
        return np.array(list(zip(local_maximums[1], local_maximums[0])))

    @timer_func
    def get_likely_balls(self):
        sum_of_frames = np.average(self.balls_images_queue, axis=2)
        # get the local maximas
        maximas = self.__get_maximas(sum_of_frames)
        if maximas is not None:
            return maximas
        return np.array([])


class BallDetector:
    def __init__(self, back_fiducial_id, front_fiducial_id, buffer_size=10):
        self.fiducial_detector = CuestickDetector(back_fiducial_id=back_fiducial_id,
                                                  front_fiducial_id=front_fiducial_id)
        self.balls_buffer = BallBuffer(buffer_size)

    @timer_func
    def detect_balls(self, image):

        # remove the fiducials
        image = remove_fiducials(image, self.fiducial_detector.back_fiducial_id,
                                 self.fiducial_detector.front_fiducial_id)
        # Remove green from the image
        no_green_ = remove_green(image, new_color=Colors.GREEN)
        # Apply bilateral filter
        bilat_ = bilateral_filter(no_green_)
        # Apply canny edge detection
        canny_ = canny_edge_detection(bilat_)
        # Apply hough circles
        circles = hough_circles(canny_, 15, 30, 20, 50, 30)

        circles = self.change_circles_format(circles)

        # find the black ball seperately:
        try:
            white_ball = find_white_ball(image, white_ball_temp)
            if white_ball is not None:
                circles = np.vstack([circles, white_ball])
            black_ball = find_black_ball(image, black_ball_temp)
            if black_ball is not None:
                circles = np.vstack([circles, black_ball])
        except:
            pass
        # if circles is not empty:
        if circles is not None:
            circles = np.unique(circles, axis=0)
            self.balls_buffer.add_balls(circles)
        return self.balls_buffer.get_likely_balls()

    def change_circles_format(self, circles):
        if circles is not None:
            circles = np.int32(circles[0, :][:, :2])
        return circles


if __name__ == '__main__':
    # load the video
    cap = cv2.VideoCapture("detect_objects_test_images/newest_test_video.mp4")
    rect = [(36, 931), (60, 79), (1754, 108), (1735, 970)]

    balls_finder = BallDetector(3, 4, buffer_size=10)
    ret, frame = cap.read()
    # save this frame
    for i in range(500):
        ret, frame = cap.read()
        if not ret:
            break
        cropped_image = transform_board(frame, rect)
        cv2.imwrite("balls_for_template_matching/first_frame.jpg", cropped_image)
        # cropped_image = frame
        likely_balls = balls_finder.detect_balls(cropped_image)
        image = cropped_image.copy()
        for ball in likely_balls:
            cv2.circle(image, ball, 30, (0, 255, 0), 3)

        # draw the black and white balls
        black_ball = find_black_ball(cropped_image, black_ball_temp)
        white_ball = find_white_ball(cropped_image, white_ball_temp)
        cv2.circle(image, black_ball, 30, (0, 0, 0), 3)
        cv2.circle(image, white_ball, 30, (255, 255, 255), 3)

        print(f"new frame {i}")
        # resize

        image = cv2.resize(image, (400, 800))
        cv2.imshow("balls", image)
        cropped_copy = copy.deepcopy(cropped_image)
        if cv2.waitKey(1) & 0xFF == ord('d'):
            # show the intermediates
            intermediates = draw_intermediate_images(bilateral_filter(remove_green(cropped_image, new_color=Colors.BLACK)),
                                        canny_edge_detection(bilateral_filter(remove_green(cropped_image, new_color=Colors.BLACK))),
                                        cropped_copy, remove_green(cropped_image, new_color=Colors.BLACK))
            show_im(intermediates)
            continue

        if cv2.waitKey(1) & 0xFF == ord('s'):
            for j in range (10):
                ret, frame = cap.read()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
#
#
# if __name__ == '__main__':
#
#     from perfectswish.utils import webcam
#     detector = CuestickDetector(back_fiducial_id=3, front_fiducial_id=4)
#     cap = cv2.VideoCapture("detect_objects_test_images/fiducials_2.mp4")
#
#     buffer = FrameBuffer(5)
#
#     while True:
#         ret, frame = cap.read()
#         rect = [(36, 931), (60, 79), (1754, 108), (1735, 970)]
#
#         cropped_image = transform_board(frame, rect)
#         # cover the aruco markers
#         detector.detect_cuestick(cropped_image)
#         cropped_image = detector.cover_aruco_markers(cropped_image)
#
#         buffer.add_frame(cropped_image)
#         average_frame = buffer.get_average_frame()
#         average_frame = cropped_image
#         if average_frame is not None:
#
#             intermediate, circles = find_balls(average_frame, new_color=[0, 255, 0], return_intermediates=True)
#             if circles is not None:
#                 drawn = draw_circles(average_frame, circles)
#                 drawn = cv2.resize(drawn, (800, 600))
#                 cv2.imshow("average_frame", drawn)
#             # shrink intermediate to fit the screen
#             cv2.imshow("black_and_white_only", cv2.resize(draw_black_and_white_balls(average_frame), (800, 600)))
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
