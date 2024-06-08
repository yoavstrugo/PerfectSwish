import cv2
import numpy as np
from screeninfo import Monitor

from perfectswish.utils.utils import Colors
from perfectswish.utils.webcam import WebcamCapture
from perfectswish.image_transformation.image_processing import Board, transform_board
from perfectswish.new_image_transformation.base_image_frame import BaseImageFrame
from perfectswish.new_image_transformation.display_app import DisplayApp
from perfectswish.new_image_transformation.image_transform_frame_decorator import ImageTransform
from perfectswish.new_image_transformation.phase import Phase
from perfectswish.new_image_transformation.points_frame_decorator import Points
from perfectswish.object_detection.detect_balls import draw_circles, find_balls
from perfectswish.object_detection.detect_cuestick import CuestickDetector
from perfectswish.visualization.visualize import draw_board


RESOLUTION_FACTOR = 8

class GamePhase(Phase):
    def __init__(self, app, crop_rect, project_rect, cap: WebcamCapture, other_screen: Monitor,
                 next_func, back_func):
        super().__init__('game')

        self.__crop_rect = crop_rect
        self.__project_rect = project_rect
        self.__cap = cap
        self.__cue_detector = CuestickDetector()

        self.__frame = ImageTransform(
            Points(
                BaseImageFrame(app, app, cap.get_latest_image),
                initial_points=project_rect
            )
        )
        self.__top_lvl = DisplayApp(other_screen)
        projection = ImageTransform(
            Points(
                BaseImageFrame(self.__top_lvl, app, cap.get_latest_image, width=other_screen.width,
                               height=other_screen.height),
                initial_points=project_rect
            )
        )
        self.__top_lvl.set_frame(projection)
        self.__frame.pack(fill="both", expand=True)

        # start a new proccess with main_loop, it shall share a variable with the main process. main_loop
        # would update the shared variable with the image to be displayed. The main process would read the
        # shared variable and display the image on the other screen:
        # self.__manager = Manager()
        # self.__balls_list = self.__manager.list()
        # self.__balls_process = Process(target=self.__process_balls, args=(self.__balls_list,))

    def main_loop(self) -> np.array:
        webcam_image = self.__cap.get_latest_image()
        if not self.__crop_rect:
            return webcam_image
        cropped_board = transform_board(webcam_image, self.__crop_rect)

        balls = find_balls(cropped_board)
        cue = self.__cue_detector.detect_cuestick(cropped_board)

        board_im = draw_board(Board(112 * RESOLUTION_FACTOR, 224 * RESOLUTION_FACTOR), Colors.RED)
        if balls is not None:
            draw_circles(board_im, balls)

        if cue is not None:
            stickend, back_fiducial_center, front_fiducial_center = cue
            self.__cue_detector.draw_cuestick(board_im, stickend, back_fiducial_center, front_fiducial_center)

        return board_im

    def __process_balls(self):
        pass

    def get_data(self):
        return None

    def destroy(self):
        self.__frame.pack_forget()
        self.__top_lvl.withdraw()
        self.__top_lvl.destroy()
        self.__frame.destroy()
        # TODO: release balls process
