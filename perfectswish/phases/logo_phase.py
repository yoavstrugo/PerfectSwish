import threading
import time
from typing import Callable

import cv2

from perfectswish.new_image_transformation.display_app import DisplayApp
from perfectswish.new_image_transformation.image_transform_frame_decorator import ImageTransform
from perfectswish.utils.webcam import WebcamCapture
from perfectswish.new_image_transformation.base_image_frame import BaseImageFrame
from perfectswish.new_image_transformation.movable_points_decorator import MovablePoints
from perfectswish.new_image_transformation.phase import Phase
from perfectswish.new_image_transformation.point_selection_frame_decorator import PointSelection
from perfectswish.new_image_transformation.points_frame_decorator import Points
from perfectswish.new_image_transformation.user_action_frame import UserActionFrame


class LogoPhase(Phase):
    """
    The crop phase. This will allow the user to crop the image.
    """
    def __init__(self, app: "PerfectSwishApp", project_rect, other_screen, next_func: Callable, logo_path):
        """
        Initializes the crop phase.
        :param app: the main app.
        :param saved_data: the saved data to load.
        :param next_func: the next function.
        :param back_func: the back function.
        :param cap:
        """
        super().__init__("logo")
        self.__next_func = next_func

        self.__image = cv2.imread(logo_path)
        self.__stop_flag = False

        self.__top_lvl = DisplayApp(other_screen)
        projection = ImageTransform(
            Points(
                BaseImageFrame(self.__top_lvl, app, self.__get_logo, width=other_screen.width,
                               height=other_screen.height),
                initial_points=project_rect,
                points_in_relation_to=(1920, 1080)
            )
        )
        self.__top_lvl.set_frame(projection)
        self.__logo_timer_thread = threading.Thread(target=self.__logo_timer_proc)
        self.__logo_timer_thread.start()

    def __get_logo(self):
        if self.__stop_flag:
            self.__next_func()
        return self.__image

    def __logo_timer_proc(self):
        time.sleep(5)
        self.__stop_flag = True

    def get_data(self):
        pass

    def destroy(self):
        self.__top_lvl.withdraw()
        self.__top_lvl.destroy()
        self.__logo_timer_thread.join()

