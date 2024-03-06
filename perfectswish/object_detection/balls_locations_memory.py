import cv2
import numpy as np
from perfectswish.api.common_objects import Ball
from typing import Union, List, Tuple
import os
from tqdm import tqdm as wassach
from perfectswish.object_detection.detect_objects import ball_objects


def return_balls_list(recent_balls_list: List[List[Ball]], id_to_ball: dict[float, Ball], last_frame_num=10,
                      last_frame_percent=0) -> List[Ball]:
    """
    This function gets the most recent balls list and num of last frame, percent of them and return
    a list of the balls that will appear on the board
    """
    recent = recent_balls_list[-last_frame_num:]
    balls_times_dict = {}
    for frame in recent:
        for ball in frame:
            found = False
            for ball_2_id in balls_times_dict.keys():
                ball_2 = id_to_ball[ball_2_id]
                if np.linalg.norm(ball.position - ball_2.position) < 10 and ball.id != ball_2.id:
                    balls_times_dict[ball_2.id] += 1
                    found = True
                    break
            if not found:
                balls_times_dict[ball.id] = 1
    id_for_delete = []
    for id, times in balls_times_dict.items():
        if times < last_frame_percent * last_frame_num:
            id_for_delete.append(id)
    for id in id_for_delete:
        del balls_times_dict[id]
    new_balls_list = []
    for id in balls_times_dict.keys():
        new_balls_list.append(id_to_ball[id])
    return new_balls_list


def find_balls_list_and_avarage_list():
    blank_image = cv2.imread(r'images_test\balls_locations_memory_test\WIN_20240306_14_01_02_Pro.jpg')
    # iterate over the folder of the images, read all of them and run the function on each one
    folder = r'images_test\balls_locations_memory_test'
    images = [os.path.join(folder, file) for file in os.listdir(folder)]
    images.pop(0)
    images_in_cv2 = [cv2.imread(image) for image in wassach(images, desc="Reading images")]
    recent_balls_list = []

    i = 0
    id_to_ball = {}
    for image in wassach(images_in_cv2, desc="Running the function on the images"):
        balls, cue_ball = ball_objects(image, blank_image)
        for ball in balls:
            ball.id = i
            id_to_ball[i] = ball
            i += 1
        recent_balls_list.append(balls)

    balls_list_and_avarge_list = []

    i = 0
    for ball_list in wassach(recent_balls_list, desc="Running the function on the recent balls list"):
        if i < 10:
            i += 1
            continue
        balls_list_and_avarge_list.append([ball_list, return_balls_list(recent_balls_list[:i], id_to_ball, 15, 0.7)])
        i += 1

    return balls_list_and_avarge_list


# balls_list_and_avarge_list = find_balls_list_and_avarage_list()
# for i in range(len(balls_list_and_avarge_list)):
#     print(len(balls_list_and_avarge_list[i][1]))


def get_avarage_balls_list(balls_lists: List[List[Ball]]) -> List[Ball]:
    """
    This function gets a list of balls lists and return the avarge of the balls that will appear on the board
    :param balls_lists: List of balls lists, the last frames
    :return: List of balls for showing on the board
    """
    last_frame_num = len(balls_lists)
    i = 0
    id_to_ball = {}
    for frame in balls_lists:
        for ball in frame:
            ball.id = i
            id_to_ball[i] = ball
            i += 1
    showing_balls_list = return_balls_list(balls_lists, id_to_ball, last_frame_num, 0.7)
    return showing_balls_list
