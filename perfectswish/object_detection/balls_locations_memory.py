import cv2
import numpy as np
from perfectswish.api.common_objects import Ball
from typing import Union, List, Tuple
import os
from tqdm import tqdm as wassach
from perfectswish.object_detection.detect_objects import ball_objects


def return_balls_list(recent_balls_list: List[List[Ball]], id_to_ball: dict[int, Ball], last_frame_num = 10, last_frame_percent = 0) -> List[Ball]:
    """
    This function gets the 30 most recent balls list and num of last frame, percent of them and return
    a list of the balls that will appear on the board
    """
    recent = recent_balls_list[-last_frame_num:]
    balls_times_dict = {}
    for frame in recent:
        for ball in frame:
            for ball_2_id in balls_times_dict.keys():
                ball_2 = id_to_ball[ball_2_id]
                if np.linalg.norm(ball.position - ball_2.position) < 10 and ball.id != ball_2.id:
                    balls_times_dict[ball.id] += 1
                    break
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
        recent_balls_list.append(balls)

    balls_list_and_avarge_list = []

    i=0
    for ball_list in wassach(recent_balls_list, desc="Running the function on the recent balls list"):
        if i<10:
            i+=1
            continue
        balls_list_and_avarge_list.append([ball_list, return_balls_list(recent_balls_list[:i], id_to_ball, 10, 7)])
        i+=1

    return balls_list_and_avarge_list



balls_list_and_avarge_list = find_balls_list_and_avarage_list()
eighty_place = balls_list_and_avarge_list[15]


