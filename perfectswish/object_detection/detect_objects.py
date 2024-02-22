import cv2
import numpy as np
from cv2 import Mat
from typing import Union, List, Tuple
from perfectswish.api.utils import Colors
from perfectswish.api.common_objects import Ball, WhiteBall, Cue

Image = Union[Mat, np.ndarray]


def subtract_images(image_no_balls: Image, image_balls: Image, threshold: int = 80) -> Image:
    no_balls_np = np.array(image_no_balls)
    balls_np = np.array(image_balls)
    black_np = np.ones_like(no_balls_np) * 100

    absolute_diff = np.sum(np.abs(no_balls_np - balls_np), axis=2)
    mask = absolute_diff > threshold

    final_image = np.where(mask[:, :, None], balls_np, black_np)

    return final_image.astype(np.uint8)

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
    for color in contours:
        for cnt in color:
            perimeter = cv2.arcLength(cnt, True)
            area = cv2.contourArea(cnt)
            if find_circularity(cnt, perimeter, area):
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                radius = int(radius)
                add_ball = True
                for ball in balls_center_radius:
                    if np.linalg.norm(np.array(ball[0]) - np.array(center)) < 25:
                        add_ball = False
                        break
                if not add_ball:
                    continue
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


def take_threshold(bilateral_color, threshold=40):
    mask = np.any(bilateral_color > threshold, axis=-1).astype(np.uint8) * 255
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


def find_cue(ball_image, contours):
    max_length = 0
    cue_contour = None
    for cnt in contours:
        approx, length = calculateContourLength(cnt)
        if length > max_length:
            max_length = length
            cue_contour = approx
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
    if cue_ball is None:
        cue_ball = WhiteBall(balls[0].position, False, 15, Colors.WHITE, False)
    return balls, cue_ball


def create_cue_object(cue_contour, original_image, cue_ball_position):
    x1, y1, x2, y2 = cue_contour[0][0][0], cue_contour[0][0][1], cue_contour[1][0][0], cue_contour[1][0][1]
    distance_1 = np.sqrt((x1 - cue_ball_position[0]) ** 2 + (y1 - cue_ball_position[1]) ** 2)
    distance_2 = np.sqrt((x2 - cue_ball_position[0]) ** 2 + (y2 - cue_ball_position[1]) ** 2)
    if distance_1 > distance_2:
        direction = np.array([x1 - x2, y1 - y2])
        cue_edge = np.array([x1, y1])
    else:
        direction = np.array([x2 - x1, y2 - y1])
        cue_edge = np.array([x1, y1])
    cue = Cue(cue_edge, direction)
    return cue


def ball_objects(ball_center_radius, original_image: Image):
    balls, cue_ball = create_ball_objects(ball_center_radius, original_image)
    if cue_ball is None:
        return None, None
    if not balls:
        return None, None
    return balls, cue_ball


def cue_object(image_with_circles: Image, original_image: Image, contours, cue_ball: WhiteBall):
    image_with_circles_and_cue, cue_contour = find_cue(image_with_circles, contours)
    if cue_contour is None:
        return None
    cue = create_cue_object(cue_contour, original_image, cue_ball.position)
    return cue


def mask_by_colors(image: Image):
    red_mask = cv2.inRange(image, np.array([0, 0, 70]), np.array([255, 255, 255]))
    green_mask = cv2.inRange(image, np.array([0, 70, 0]), np.array([255, 255, 255]))
    blue_mask = cv2.inRange(image, np.array([70,0,0]), np.array([255, 255, 255]))
    return red_mask, green_mask, blue_mask


def find_contours(balls_image: Image, original_image: Image):
    subtracted_image = subtract_images(original_image, balls_image)
    rgb = cv2.cvtColor(subtracted_image, cv2.COLOR_BGR2RGB)
    bilateral_color = cv2.bilateralFilter(rgb, 9, 100, 20)
    red_mask, green_mask, blue_mask = mask_by_colors(bilateral_color)
    mask_image = take_threshold(bilateral_color, 40)
    masks = [red_mask, green_mask, blue_mask, mask_image]
    img_contours_array = []
    contours_array = []
    for mask in masks:
        erisioned_dilated_image = erision_dilation(mask, 3, 3)
        contours, _ = cv2.findContours(erisioned_dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_contours = np.zeros(erisioned_dilated_image.shape, dtype=np.uint8)
        cv2.drawContours(img_contours, contours, -1, Colors.WHITE, 1)
        img_contours_array.append(img_contours)
        contours_array.append(contours)

    return img_contours_array, contours_array


def find_balls_and_circle_them(contours):
    ball_center_radius, image_with_circles = find_circles(balls_image, contours)
    return ball_center_radius, image_with_circles


def find_objects(balls_image: Image, original_image: Image):
    img_contours, contours = find_contours(balls_image, original_image)
    color_contours = [contours[0], contours[1], contours[2]]
    color_img_contours = [img_contours[0], img_contours[1], img_contours[2]]
    black_img_contours = img_contours[3]
    for img_cnt in color_img_contours:
        cv2.imshow("img_cnt", img_cnt)
        cv2.waitKey(0)
    black_contours = contours[3]
    ball_center_radius, image_with_circles = find_balls_and_circle_them(color_contours)
    cv2.imshow("image_with_circles", image_with_circles)
    cv2.waitKey(0)
    balls, cue_ball = ball_objects(ball_center_radius, original_image)
    cue = cue_object(image_with_circles, original_image, black_contours, cue_ball)
    return balls, cue_ball, cue


if __name__ == '__main__':
    board_image = cv2.imread(r"C:\Users\TLP-299\PycharmProjects\PerfectSwish\perfectswish\image_transformation\images\WIN_20240222_09_04_29_Pro.jpg")
    balls_image = cv2.imread(r"C:\Users\TLP-299\PycharmProjects\PerfectSwish\perfectswish\image_transformation\images\WIN_20240222_09_05_19_Pro.jpg")
    board_bilateral = cv2.bilateralFilter(board_image, 9, 100, 20)
    #cv2.imshow("board", board_bilateral)
    balls_bilateral = cv2.bilateralFilter(balls_image, 9, 100, 20)
    #cv2.imshow("balls", balls_bilateral)
    subtracted = subtract_images(board_image, balls_image, threshold=250)
    cv2.imshow("Subtracted images", subtracted)
    medianed = cv2.medianBlur(subtracted, 3)
    cv2.imshow("medianed", medianed)
    cv2.waitKey(0)
