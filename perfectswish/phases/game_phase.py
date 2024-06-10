import multiprocessing
import threading
import time
from multiprocessing import Manager, Process
from typing import NoReturn

import numpy as np
from screeninfo import Monitor

from perfectswish.image_transformation.image_processing import Board, transform_board
from perfectswish.new_image_transformation.base_image_frame import BaseImageFrame
from perfectswish.new_image_transformation.display_app import DisplayApp
from perfectswish.new_image_transformation.image_transform_frame_decorator import ImageTransform
from perfectswish.new_image_transformation.phase import Phase
from perfectswish.new_image_transformation.points_frame_decorator import Points
from perfectswish.object_detection.detect_balls import draw_circles, find_balls
from perfectswish.object_detection.detect_cuestick import CuestickDetector
from perfectswish.settings import BOARD_BASE_HEIGHT, BOARD_BASE_WIDTH, BOARD_SIZE, RESOLUTION_FACTOR
from perfectswish.utils.utils import Colors
from perfectswish.utils.webcam import MultiprocessWebcamCapture
from perfectswish.visualization.visualize import draw_board


class GamePhase(Phase):
    """
    The game phase, where the game is played. This will project the 'assist' screen on the other screen,
    and the control screen on the main screen.
    """

    # The empty board image
    __EMPTY_BOARD_IMAGE = np.zeros((BOARD_SIZE.height, BOARD_SIZE.width, 3), dtype=np.uint8)

    def __init__(self, app: "PerfectSwish", crop_rect: np.ndarray, project_rect: np.ndarray,
                 cap: MultiprocessWebcamCapture,  other_screen: Monitor, fps: int,
                 balls_update_rate: int):
        """
        Initializes the game phase.
        :param app: The main app.
        :param crop_rect: The crop rectangle.
        :param project_rect: The project rectangle.
        :param cap: The webcam capture.
        :param other_screen: The other screen, to project on.
        :param fps: The fps in which to update the projection.
        :param balls_update_rate: The rate in which to update the balls processing.
        """

        super().__init__('game')

        self.__crop_rect = crop_rect
        self.__project_rect = project_rect
        self.__cap = cap
        self.__cue_detector = CuestickDetector()

        self.__fps = fps
        self.__balls_update_rate = balls_update_rate

        self.__next_image = None
        self.__real_board_img = None
        self.__real_board_img_lock = threading.Lock()

        # start a new proccess with main_loop, it shall share a variable with the main process. main_loop
        # would update the shared variable with the image to be displayed. The main process would read the
        # shared variable and display the image on the other screen:
        self.__manager = Manager()
        self.__stop_event = multiprocessing.Event()
        self.__shared_balls_list_lock = multiprocessing.Lock()
        self.__shared_balls_list = self.__manager.list()

        self.__output_image_lock = threading.Lock()
        self.__output_image_thread = threading.Thread(target=self.__generate_output_image, daemon=True)
        self.__output_image_thread.start()

        # it takes params: stop_event, cap, crop_rect, lock, shared_balls_list, update_rate
        self.__balls_process = Process(name='Balls Find', target=balls_update_process, args=(
            self.__stop_event,
            self.__cap,
            self.__crop_rect,
            self.__shared_balls_list_lock,
            self.__shared_balls_list,
            self.__balls_update_rate))
        self.__balls_process.start()

        self.__frames = list()
        # let height be the height of the window
        board_height = app.winfo_height()
        # let the width be such that it keeps proportions
        board_width = int(board_height * (BOARD_SIZE.width / BOARD_SIZE.height))
        self.__frames.append(
            BaseImageFrame(app, app, self.__get_real_board, width=board_width,
                           height=board_height)
        )

        self.__top_lvl = DisplayApp(other_screen)
        projection = ImageTransform(
            Points(
                BaseImageFrame(self.__top_lvl, app, self.__get_next_image, width=other_screen.width,
                               height=other_screen.height),
                initial_points=project_rect,
                points_in_relation_to=(1920, 1080)
            )
        )
        self.__top_lvl.set_frame(projection)
        for frame in self.__frames:
            if frame is not None:
                frame.pack()

    def __get_real_board(self) -> np.ndarray:
        """
        Returns the real board image, multiprocess safe.
        """
        with self.__real_board_img_lock:
            if self.__real_board_img is None:
                return self.__EMPTY_BOARD_IMAGE
            return self.__real_board_img

    def __get_next_image(self) -> np.ndarray:
        """
        Returns the next image to be displayed (the assiting image), multiprocess safe.
        """
        with self.__output_image_lock:
            if self.__next_image is None:
                return self.__EMPTY_BOARD_IMAGE
            return self.__next_image

    def __write_next_image(self, im: np.ndarray) -> None:
        """
        Writes the next image to be displayed. multiprocess safe.
        :param im: the image
        """
        with self.__output_image_lock:
            self.__next_image = im

    def __generate_output_image(self) -> NoReturn:
        """
        This will continously generate the output image.
        """
        while not self.__stop_event.is_set():
            webcam_image = self.__cap.get_latest_image()
            if not self.__crop_rect:
                raise ValueError("Crop rect not set")
            cropped_board = transform_board(webcam_image, self.__crop_rect)

            balls = self.__read_balls()
            cue = self.__cue_detector.detect_cuestick(cropped_board)

            board_im = draw_board(
                Board(width=BOARD_BASE_WIDTH * RESOLUTION_FACTOR,
                      height=BOARD_BASE_HEIGHT * RESOLUTION_FACTOR),
                Colors.RED)
            if balls is not None:
                draw_circles(board_im, balls)
                draw_circles(cropped_board, balls)

            if cue is not None:
                stickend, back_fiducial_center, front_fiducial_center = cue
                self.__cue_detector.draw_cuestick(board_im, stickend, back_fiducial_center,
                                                  front_fiducial_center)
                self.__cue_detector.draw_cuestick(cropped_board, stickend, back_fiducial_center,
                                                  front_fiducial_center)

            with self.__real_board_img_lock:
                self.__real_board_img = cropped_board

            self.__write_next_image(board_im)
            time.sleep(1 / self.__fps)

    def __write_balls(self, balls: np.ndarray) -> None:
        """
        Writes the balls np.ndarray into the shared balls list, with respect to the global lock.
        """
        balls_list: list = balls.tolist()
        with self.__shared_balls_list_lock:
            self.__shared_balls_list[:] = []
            self.__shared_balls_list.extend(balls_list)

    def __read_balls(self) -> np.ndarray:
        """
        Reads the shared balls list and returns it as a np.ndarray. With respect to the global lock.
        """
        with self.__shared_balls_list_lock:
            balls_ndarray = np.array(self.__shared_balls_list)
            return balls_ndarray

    def get_data(self):
        return None

    def destroy(self) -> None:
        """
        Destroys the game phase. and releases all resources.
        """
        for frame in self.__frames:
            if frame is not None:
                frame.pack_forget()
                frame.destroy()
        self.__top_lvl.withdraw()
        self.__top_lvl.destroy()
        self.__stop_event.set()
        self.__balls_process.join()
        self.__manager.shutdown()


def balls_update_process(stop_event, cap, crop_rect, lock, shared_balls_list, update_rate) -> NoReturn:
    """
    The ball update process entry point, will run at rate of update_rate
    """
    while not stop_event.is_set():
        frame = cap.get_latest_image()
        cropped = transform_board(frame, crop_rect)
        balls: np.ndarray = find_balls(cropped)
        balls_list = balls.tolist()
        with lock:
            shared_balls_list[:] = []
            shared_balls_list.extend(balls_list)

        time.sleep(1 / update_rate)
