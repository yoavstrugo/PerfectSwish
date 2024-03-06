import cv2
import numpy as np
from cv2 import Mat
from typing import Union, List, Tuple
from perfectswish.api.utils import Colors
from perfectswish.api.common_objects import Ball, WhiteBall, Cue

Image = Union[Mat, np.ndarray]


def subtract_images(image1: Image, image2: Image) -> Image:
    image2_with_neg = image1.astype(np.int32)
    image1_with_neg = image2.astype(np.int32)
    return np.abs(image2_with_neg - image1_with_neg).astype(np.uint8)


def find_circularity(cnt, perimeter, area, circularity_thersold=0.3):
    if perimeter == 0:
        return False
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    if circularity_thersold < circularity <= 1.6:
        return True
    return False



def find_circles(balls_image, contours, min_radius=9, max_radius=25):
    balls_center_radius = []
    image_with_circles = balls_image.copy()
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        if find_circularity(cnt, perimeter, area):
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            if radius > min_radius and radius < max_radius:
                cv2.circle(image_with_circles, center, radius, (0, 255, 0), 2)
                ball = (center, radius)
                balls_center_radius.append(ball)
    return balls_center_radius, image_with_circles


def erision_dilation(mask, kernel_e_num: int, kernel_d_num: int):
    kernel_e = np.ones((kernel_e_num, kernel_e_num), np.uint8)
    kernel_d = np.ones((kernel_d_num, kernel_d_num), np.uint8)
    mask = cv2.erode(mask, kernel_e, iterations=1)
    mask = cv2.dilate(mask, kernel_d, iterations=1)
    return mask


def take_threshold(bilateral_color, threshold=40, balls_image=None):
    mask = np.any(bilateral_color > threshold, axis=-1).astype(np.uint8) * 255
    # every pixel that is black in the mask image, will be black in the balls image
    balls_image_subtracted = cv2.bitwise_and(balls_image, balls_image, mask=mask)
    #show the image but small window, without cut it
    cv2.imshow("balls_image_subtracted", cv2.resize(balls_image_subtracted, (0, 0), fx=0.5, fy=0.5))
    return mask


def images_formats(image: Image) -> Tuple[Image, Image, Image, Image]:
    """
    param1: image
    return: Tuple[Image, Image, Image, Image]
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bilateral_color = cv2.bilateralFilter(rgb, 9, 100, 20)
    gray = cv2.cvtColor(bilateral_color, cv2.COLOR_BGR2GRAY)
    return rgb, hsv, bilateral_color, gray


def find_linear_lines(contours, img_contours):
    lines = cv2.HoughLinesP(img_contours, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    if lines is None:
        return [], img_contours
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_contours, (x1, y1), (x2, y2), (255, 0, 0), 3)
    return lines, img_contours

def find_cue_2(ball_image, img_contours, contours):
    lines, img_contours = find_linear_lines(contours, img_contours)
    cv2.waitKey(0)
    #find the longest line and return the two points of the line, and draw it on the images
    max_length = 0
    cue_contour = None
    if lines is None:
        return ball_image, cue_contour
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if (y1<70 or y1>1000) or (y2<70 or y2>1000) or (x1<70 or x1>1800) or (x2<70 or x2>1800):
            if np.arctan2(y2 - y1, x2 - x1) < 0.1 or np.arctan2(y2 - y1, x2 - x1) > 1.5:
                continue
        length = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        if length > max_length:
            max_length = length
            cue_contour = np.array([[x1, y1], [x2, y2]])
    if cue_contour is not None:
        cv2.drawContours(ball_image, [cue_contour], -1, Colors.GREEN, 3)
    return ball_image, cue_contour





def calculateContourLength(cnt):
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    length = cv2.arcLength(approx, False)
    return approx, length


def find_lines(img_contours, image_with_circles):
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 15  # minimum number of pixels making up a line
    max_line_gap = 500  # maximum gap in pixels between connectable line segments
    # creating a blank to draw lines on
    line_image = np.copy(img_contours) * 0  # creating a blank to draw lines on
    lines = cv2.HoughLinesP(img_contours, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    lines = [line for line in lines if abs(line[0][0] - line[0][2]) > 20 and abs(line[0][1] - line[0][3]) > 20]
    filtered_lines = []
    if lines is None:
        return filtered_lines
    for line in lines:
        if not filtered_lines:
            filtered_lines.append(line)
            continue
        for line_2 in filtered_lines:
            if (abs(line[0][0] - line_2[0][0]) < 10 and abs(line[0][2] - line_2[0][2]) < 10) or (
                    abs(line[0][1] - line_2[0][1]) < 10 and abs(line[0][3] - line_2[0][3]) < 10):
                break
    for line in filtered_lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image_with_circles, (x1, y1), (x2, y2), (255, 0, 0), 5)

    return filtered_lines, image_with_circles


def find_ball_color(image, center, radius):
    """
    #     WHITE = (255, 255, 255)
    #     YELLOW = (0, 255, 255)
    #     BLUE = (255,0,0)
    #     RED = (0,0,255)
    #     PURPLE = (255,0,255)
    #     ORANGE = (0,165,255)
    #     GREEN = (0,255,0)
    #     BROWN = (42,42,165)
    #     BLACK = (0,0,0)
    """
    sum_white = 0
    sum_red = 0
    sum_green = 0
    sum_blue = 0
    counter = 0
    stripped = False
    radius = 5
    sum_red = image[center[1]][center[0]][2]
    sum_green = image[center[1]][center[0]][1]
    sum_blue = image[center[1]][center[0]][0]

    if sum_red > 200 and sum_green > 200 and sum_blue > 200:
        return Colors.WHITE, False
    if sum_white > 50:
        stripped = True
    if sum_red > 130 and sum_green < 100 and sum_blue < 100:
        return Colors.RED, stripped
    if sum_red < 100 and sum_green > 130 and sum_blue < 100:
        return Colors.GREEN, stripped
    if sum_red < 100 and sum_green < 100 and sum_blue > 130:
        return Colors.BLUE, stripped
    if sum_red > 130 and sum_green > 130 and sum_blue < 100:
        return Colors.YELLOW, stripped
    if sum_red > 100 and sum_green < 100 and sum_blue > 130:
        return Colors.PURPLE, stripped
    if sum_red > 130 and sum_green > 100 and sum_blue < 100:
        return Colors.ORANGE, stripped
    if sum_red < 100 and sum_green < 100 and sum_blue < 100:
        return Colors.BLACK, stripped
    if sum_red < 130 and sum_green < 130 and sum_blue < 130:
        return Colors.BROWN, stripped
    return Colors.RED, stripped


def create_ball_objects(ball_center_radius, original_image):
    balls = []
    for center_radius in ball_center_radius:
        center = center_radius[0]
        radius = center_radius[1]
        color, stripped = find_ball_color(original_image, center, radius)
        ball = Ball(center, False, 15, color, False)
        balls.append(ball)
    cue_ball = None
    for ball in balls:
        if ball.color == Colors.WHITE:
            cue_ball = WhiteBall(ball.position, False, 15, ball.color, False)
            break
    if not balls:
        return None, None
    if cue_ball is None:
        cue_ball = WhiteBall(balls[0].position, False, 15, Colors.WHITE, False)

    return balls, cue_ball


def create_cue_object(cue_contour, original_image, cue_ball_position):
    x1, y1, x2, y2 = cue_contour[0][0], cue_contour[0][1], cue_contour[1][0], cue_contour[1][1]
    distance_1 = np.sqrt((x1 - cue_ball_position[0]) ** 2 + (y1 - cue_ball_position[1]) ** 2)
    distance_2 = np.sqrt((x2 - cue_ball_position[0]) ** 2 + (y2 - cue_ball_position[1]) ** 2)
    if distance_1 > distance_2:
        direction = np.array([x2-x1, y2 - y1])
        cue_edge = np.array([x2, y2])
    else:
        direction = np.array([x1 - x2, y1 - y2])
        cue_edge = np.array([x1, y1])
    cue = Cue(cue_edge, direction)
    return cue


def find_objects(balls_image: Image, original_image: Image):
    ball_center_radius, image_with_circles, img_contours, contours = find_balls(balls_image, original_image)
    image_with_circles_and_cue, cue_contour = find_cue(image_with_circles, contours)
    balls, cue_ball = create_ball_objects(ball_center_radius, original_image)
    if cue_ball is None:
        return balls, None, None
    cue = create_cue_object(cue_contour, original_image, cue_ball.position)
    return balls, cue_ball, cue


def ball_objects(balls_image: Image, original_image: Image):
    ball_center_radius, image_with_circles, img_contours, contours = find_balls(balls_image, original_image)

    balls, cue_ball = create_ball_objects(ball_center_radius, original_image)
    if cue_ball is None:
        return None, None
    if not balls:
        return None, None
    return balls, cue_ball


def cue_object(image_with_circles: Image, original_image: Image, img_contours: Image, contours, cue_ball: WhiteBall):
    image_with_circles_and_cue, cue_contour = find_cue_2(image_with_circles, img_contours, contours)
    if cue_contour is None:
        return None, image_with_circles_and_cue
    cue = create_cue_object(cue_contour, original_image, cue_ball.position)
    return cue, image_with_circles_and_cue


def find_contours(balls_image: Image, original_image: Image):
    subtracted_image = subtract_images(original_image, balls_image)
    rgb = cv2.cvtColor(subtracted_image, cv2.COLOR_BGR2RGB)
    bilateral_color = cv2.bilateralFilter(rgb, 9, 100, 20)
    mask_image = take_threshold(bilateral_color, 60, balls_image)
    erisioned_dilated_image = erision_dilation(mask_image, 3, 3)
    cv2.imshow("erisioned_dilated_image", erisioned_dilated_image)
    cv2.waitKey(0)
    contours, _ = cv2.findContours(erisioned_dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = np.zeros(erisioned_dilated_image.shape, dtype=np.uint8)
    img_contours_for_cue = np.zeros(erisioned_dilated_image.shape, dtype=np.uint8)
    cv2.drawContours(img_contours, contours, -1, Colors.WHITE, 1)
    cv2.drawContours(img_contours_for_cue, contours, -1, Colors.WHITE, 1)
    #blur much the imgae for cue and threshold that will make the lines more fat
    img_contours_for_cue = cv2.GaussianBlur(img_contours_for_cue, (45,45), 0)

    img_contours_for_cue = cv2.threshold(img_contours_for_cue, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    img_contours_for_cue = cv2.dilate(img_contours_for_cue, np.ones((4, 4), np.uint8), iterations=1)
    return img_contours, contours, img_contours_for_cue


def find_balls_and_circle_them(contours, balls_image: Image):
    ball_center_radius, image_with_circles = find_circles(balls_image, contours)
    return ball_center_radius, image_with_circles


def find_objects(balls_image: Image, original_image: Image, TEST = False):
    cv2.imshow("balls_image", balls_image)
    cv2.waitKey(0)
    img_contours, contours, img_contours_for_cue = find_contours(balls_image, original_image)
    cv2.imshow("img_contours", img_contours)
    cv2.waitKey(0)
    ball_center_radius, image_with_circles = find_balls_and_circle_them(contours, balls_image)
    cv2.imshow("image_with_circles", image_with_circles)
    cv2.waitKey(0)
    balls, cue_ball = ball_objects(ball_center_radius, original_image)
    cue, image_with_circles_and_cue = cue_object(image_with_circles, original_image, img_contours_for_cue, contours, cue_ball)
    #show the image but in small window, without cut it
    image_with_circles_and_cue = cv2.resize(image_with_circles_and_cue, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("image_with_circles_and_cue", image_with_circles_and_cue)


    cv2.waitKey(0)
    if TEST:
        return image_with_circles_and_cue
    return balls, cue_ball, cue

def return_gradient_by_color(balls_image):
    b, g, r = cv2.split(balls_image)

    # Apply Sobel operator to each channel
    sobelx_b = cv2.Sobel(b, cv2.CV_64F, 1, 0, ksize=5)
    sobelx_g = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=5)
    sobelx_r = cv2.Sobel(r, cv2.CV_64F, 1, 0, ksize=5)
    sobely_b = cv2.Sobel(b, cv2.CV_64F, 0, 1, ksize=5)
    sobely_g = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=5)
    sobely_r = cv2.Sobel(r, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude_b = np.sqrt(sobelx_b ** 2 + sobely_b ** 2)
    gradient_magnitude_g = np.sqrt(sobelx_g ** 2 + sobely_g ** 2)
    gradient_magnitude_r = np.sqrt(sobelx_r ** 2 + sobely_r ** 2)
    gradient_magnitude = np.sqrt(gradient_magnitude_b ** 2 + gradient_magnitude_g ** 2 + gradient_magnitude_r ** 2)
    cv2.imshow("gradient_magnitude", gradient_magnitude)
    cv2.waitKey(0)


if __name__ == '__main__':
    board_image = cv2.imread(r"detect_objects_test_images\blank.jpg")
    balls_image = cv2.imread(r"detect_objects_test_images\triangle_without_triangle.jpg")
    find_objects(balls_image, board_image)
