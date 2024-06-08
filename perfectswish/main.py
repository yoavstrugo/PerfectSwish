import tkinter as tk
from typing import List

import cv2
import numpy as np
from perfectswish.object_detection.balls_locations_memory import get_avarage_balls_list
from perfectswish.object_detection.detect_objects import find_objects

from image_transformation.image_processing import find_board, generate_projection, transform_board
from perfectswish.api import webcam
from perfectswish.api.common_objects import Ball, Board, VelocityVector
from perfectswish.api.data_io import load_data, save_data
from perfectswish.gui.live_image_display import LiveImageDisplay
from perfectswish.gui.take_image_gui import take_image
from perfectswish.image_transformation.gui_crop import get_camera_rect
from perfectswish.image_transformation.gui_projection import get_projection_rect
from perfectswish.object_detection.detect_balls import draw_circles, find_balls
from perfectswish.object_detection.detect_cuestick import CuestickDetector
from perfectswish.simulation import simulate
from perfectswish.visualization.visualize import draw_board, visualize_pool_board

DATA_DIRECTORY = "data"


def create_data_directory():
    """
    Create the data directory if it does not exist.
    :return:
    """
    import os
    if not os.path.exists(DATA_DIRECTORY):
        os.makedirs(DATA_DIRECTORY)


def save_rects(camera_rect: list, projector_rect: list):
    """
    Save the camera and projector rects to a file.
    :param camera_rect:
    :param projector_rect:
    :return:
    """
    np_camera_rect = np.array(camera_rect)
    np_projector_rect = np.array(projector_rect)

    create_data_directory()

    save_data("camera_rect.npy", np_camera_rect)
    save_data("projector_rect.npy", np_projector_rect)


def load_rects() -> tuple:
    """
    Load the camera and projector rects from a file.
    :return:
    """
    try:
        np_camera_rect = load_data("camera_rect.npy")
        np_projector_rect = load_data("projector_rect.npy")
    except FileNotFoundError:
        return None, None

    camera_rect = np_camera_rect.tolist()
    projector_rect = np_projector_rect.tolist()

    return camera_rect, projector_rect


def setup_rects(cap: cv2.VideoCapture) -> tuple:
    """
    Setup the camera and projector rects.
    :param cap:
    :return:
    """
    initial_camera_rect, initial_projector_rect = load_rects()

    camera_rect = get_camera_rect(get_board_image(cap), initial_camera_rect)
    projector_rect = get_projection_rect(get_board_image(cap), initial_projector_rect)

    save_rects(camera_rect, projector_rect)

    return camera_rect, projector_rect


cap = webcam.initialize_webcam(1)
cuestick_detector = CuestickDetector()

RESOLUTION_FACTOR = 8



    # path, hit = simulate.generate_path(white_ball=cue_ball, balls=balls, board=board, max_len=10)
    #
    # direction_vectors = [
    #     VelocityVector(hit.white_ball_pos, hit.white_ball_hit_vec),
    #     VelocityVector(hit.hit_point, hit.ball_hit_vec),
    # ]
    #
    # visualized_image = visualize_pool_board(board, path=path, direction_vectors=direction_vectors,
    #                                         balls=balls)
    #
    # projection = generate_projection(visualized_image, projector_rect)
    # return board_im


def create_controls(root, return_values, cap):
    def set_image():
        return_values['image'] = webcam.get_webcam_image(cap)
        root.destroy()

    button = tk.Button(root, text="Save Empty Board", command=lambda: set_image())
    button.pack(side=tk.RIGHT, padx=10, pady=10)

    root.bind("<s>", lambda event: set_image())

    canvas = tk.Canvas(root, width=900, height=1080)
    canvas.pack(side=tk.LEFT, padx=10, pady=10)

    return canvas, [button]


def get_board_image(cap):
    return webcam.get_webcam_image(cap)


def get_empty_image(cap):
    return take_image(cap, "Take image with empty table")


def main():
    try:
        cap = webcam.initialize_webcam(index=1)
        camera_rect, projector_rect = setup_rects(cap)
        empty_image = get_empty_image(cap)
        empty_image_transformed = transform_board(empty_image, camera_rect)
    except Exception as e:
        print("Error setting up camera and projector")
        print(e)
        return

    balls_list = []

    def add_to_balls_list(balls):
        nonlocal balls_list
        balls_list = remember_balls(balls, balls_list)
        return balls_list

    liveImageDisplay = LiveImageDisplay(main_loop, cap, camera_rect, projector_rect, empty_image_transformed,
                                        add_to_balls_list,
                                        window_name="Perfect Swish", framerate=30, display_last_image=True,
                                        borderless=True,
                                        height=1080, width=1920, display_on_second_monitor=True)

    liveImageDisplay.run()


def remember_balls(balls: List[Ball], balls_list: List[List[Ball]]):
    """
    Remember the balls in the balls list.
    :param balls: The balls to remember.
    :param balls_list: The list of balls lists to remember the balls in.
    :return:
    """
    balls_list.append(balls)
    if len(balls_list) > 10:
        balls_list.pop(0)
    return get_avarage_balls_list(balls_list)


if __name__ == '__main__':
    main()
