import multiprocessing
import threading
import time
from multiprocessing import Manager, Process
from typing import NoReturn

from skimage.draw import draw

from perfectswish.settings import HOLE_RADIUS
import cv2
import numpy as np
from screeninfo import Monitor
from perfectswish.image_transformation.image_processing import Board, transform_board, transform_cue
from perfectswish.new_image_transformation.base_image_frame import BaseImageFrame
from perfectswish.new_image_transformation.display_app import DisplayApp
from perfectswish.new_image_transformation.image_transform_frame_decorator import ImageTransform
from perfectswish.new_image_transformation.phase import Phase
from perfectswish.new_image_transformation.points_frame_decorator import Points
from perfectswish.object_detection.detect_balls import draw_circles, find_balls, BallDetector
from perfectswish.object_detection.detect_cuestick import CuestickDetector
from perfectswish.settings import BOARD_BASE_HEIGHT, BOARD_BASE_WIDTH, BOARD_SIZE, RESOLUTION_FACTOR
from perfectswish.simulation import simulate
from perfectswish.simulation.simulate import normalize
from perfectswish.utils.utils import Colors
from perfectswish.utils.webcam import MultiprocessWebcamCapture
from perfectswish.visualization.visualize import draw_board
from perfectswish.simulation.simulate import REAL_BALL_RADIUS_PIXELS

STICKEND_VALUE = 10
AVERAGE = True
FRONT_FIDUCIAL_ID = 4
STICKEND_RADIUS = 10
BACK_FIDUCIAL_ID = 3


class GamePhase(Phase):
    """
    The game phase, where the game is played. This will project the 'assist' screen on the other screen,
    and the control screen on the main screen.
    """

    # The empty board image
    __EMPTY_BOARD_IMAGE = np.zeros((BOARD_SIZE.height, BOARD_SIZE.width, 3), dtype=np.uint8)

    def __init__(self, app: "PerfectSwish", crop_rect: np.ndarray, project_rect: np.ndarray,
                 cap: MultiprocessWebcamCapture, other_screen: Monitor, fps: int,
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
        self.__cue_detector = CuestickDetector(back_fiducial_id=3, front_fiducial_id=4,
                                               fiducial_to_stickend_ratio=2 / 3)

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

    def __find_max(self, last_list):
        # maximum
        max_dict = {}
        for i in range(len(last_list)):
            mode = False
            for key in max_dict.keys():
                if np.linalg.norm(last_list[i] - key) < STICKEND_RADIUS:
                    max_dict[tuple(key)] += 1
                    mode = True
            if not mode:
                max_dict[tuple(last_list[i])] = 1
        max_key = max(max_dict, key=max_dict.get)
        # to ndarray
        max_key = np.array(max_key)
        return max_key

    def __generate_output_image(self, SHOW_HOLES=False) -> NoReturn:
        """
        This will continously generate the output image.
        """
        last_stickend = []
        last_back_fiducial_center = []
        last_front_fiducial_center = []

        while not self.__stop_event.is_set():
            webcam_image = self.__cap.get_latest_image()
            if not self.__crop_rect:
                raise ValueError("Crop rect not set")
            stickend, back_fiducial_center, front_fiducial_center, marker_corners, marker_IDs = self.__cue_detector.detect_cuestick(
                webcam_image, debug=True)
            cropped_board, stickend, back_fiducial_center, front_fiducial_center = transform_cue(webcam_image,
                                                                                                 self.__crop_rect,
                                                                                                 stickend,
                                                                                                 back_fiducial_center,
                                                                                                 front_fiducial_center)
            # self.__cue_detector.back_fiducial_center_coords = back_fiducial_center
            # self.__cue_detector.front_fiducial_center_coords = front_fiducial_center

            # average
            if len(last_stickend) < STICKEND_VALUE:
                last_stickend.append(stickend)
                last_back_fiducial_center.append(back_fiducial_center)
                last_front_fiducial_center.append(front_fiducial_center)
            else:
                last_stickend.pop(0)
                last_stickend.append(stickend)
                last_back_fiducial_center.pop(0)
                last_back_fiducial_center.append(back_fiducial_center)
                last_front_fiducial_center.pop(0)
                last_front_fiducial_center.append(front_fiducial_center)

            # average
            if last_stickend and last_stickend[0] is not None:
                stickend_ave = np.mean(last_stickend, axis=0)
                back_fiducial_center_ave = np.mean(last_back_fiducial_center, axis=0)
                front_fiducial_center_ave = np.mean(last_front_fiducial_center, axis=0)

                stickend_max = self.__find_max(tuple(last_stickend))
                back_fiducial_center_max = self.__find_max(tuple(last_back_fiducial_center))
                front_fiducial_center_max = self.__find_max(tuple(last_front_fiducial_center))

                if AVERAGE:
                    stickend = stickend_ave
                    back_fiducial_center = back_fiducial_center_ave
                    front_fiducial_center = front_fiducial_center_ave
                else:
                    stickend = stickend_max
                    back_fiducial_center = back_fiducial_center_max
                    front_fiducial_center = front_fiducial_center_max

            cuestick_exist = all(
                [stickend is not None, back_fiducial_center is not None, front_fiducial_center is not None])
            balls = self.__read_balls()
            board_im = draw_board(
                Board(width=BOARD_BASE_WIDTH * RESOLUTION_FACTOR,
                      height=BOARD_BASE_HEIGHT * RESOLUTION_FACTOR),
                Colors.GREEN)
            if balls is not None:
                # draw_circles(board_im, balls)
                draw_circles(cropped_board, balls)

            if cuestick_exist:
                self.__cue_detector.draw_cuestick(board_im, front_fiducial_center, back_fiducial_center)
                # self.__cue_detector.draw_cuestick(cropped_board)

            if cuestick_exist and balls is not None:
                target_ball = simulate.get_target_ball(balls, stickend, back_fiducial_center,
                                                       front_fiducial_center)
                if target_ball is not None:
                    non_target_balls = [ball for ball in balls if not np.array_equal(ball, target_ball)]
                    # draw the target ball with a point
                    cv2.circle(board_im, tuple(target_ball.astype(int)), 10, Colors.GREEN, 2)

                    # draw the path
                    path, ball_hit = simulate.generate_path(target_ball, non_target_balls,
                                                            front_fiducial_center - back_fiducial_center,
                                                            Board(*BOARD_SIZE), 4)

                    draw_path(board_im, path)

                    # draw the expected hit
                    if ball_hit is not None:
                        # draw the ball's collision shadow
                        cv2.circle(board_im, tuple(ball_hit.hit_point.astype(int)), REAL_BALL_RADIUS_PIXELS,
                                   Colors.WHITE, 4)

                        # draw the hit direction
                        hit_direction = normalize(ball_hit.hit_point - ball_hit.white_ball_pos)
                        arrow_start = ball_hit.hit_point + hit_direction * 2 * REAL_BALL_RADIUS_PIXELS
                        arrow_end = arrow_start + hit_direction * 250
                        cv2.arrowedLine(board_im, tuple(arrow_start.astype(int)),
                                        tuple(arrow_end.astype(int)), Colors.GREEN, 7)

            for ball in balls:
                cv2.circle(board_im, tuple(ball.astype(int)), REAL_BALL_RADIUS_PIXELS + 10, Colors.GREEN, 3)

            if SHOW_HOLES:
                holes = [(0, 0), (0, BOARD_BASE_HEIGHT * RESOLUTION_FACTOR), (BOARD_BASE_WIDTH * RESOLUTION_FACTOR, 0),
                         (BOARD_BASE_WIDTH * RESOLUTION_FACTOR, BOARD_BASE_HEIGHT * RESOLUTION_FACTOR),
                         (0, BOARD_BASE_HEIGHT * RESOLUTION_FACTOR / 2),
                         (BOARD_BASE_WIDTH * RESOLUTION_FACTOR, BOARD_BASE_HEIGHT * RESOLUTION_FACTOR / 2)]
                for hole in holes:
                    hole_int = tuple(np.int32(hole))
                    cv2.circle(board_im, hole_int, HOLE_RADIUS, Colors.GREEN, -1)

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


def balls_update_process(stop_event, cap, crop_rect, lock, shared_balls_list, update_rate) -> (
        NoReturn):
    """
    The ball update process entry point, will run at rate of update_rate
    """
    balls_detector = BallDetector(back_fiducial_id=BACK_FIDUCIAL_ID, front_fiducial_id=FRONT_FIDUCIAL_ID,
                                  buffer_size=10)
    while not stop_event.is_set():
        frame = cap.get_latest_image()
        cropped = transform_board(frame, crop_rect)
        balls: np.ndarray = balls_detector.detect_balls(cropped)
        if balls is not None:
            balls_list = balls.tolist()
        else:
            balls_list = []
        with lock:
            shared_balls_list[:] = []
            shared_balls_list.extend(balls_list)

        time.sleep(1 / update_rate)


def trapez(y, y0, w):
    return np.clip(np.minimum(y + 1 + w / 2 - y0, -y + 1 + w / 2 + y0), 0, 1)


def weighted_line(r0, c0, r1, c1, w, rmin=0, rmax=np.inf, negative_order=False):
    # The algorithm below works fine if c1 >= c0 and c1-c0 >= abs(r1-r0).
    # If either of these cases are violated, do some switches.
    if abs(c1 - c0) < abs(r1 - r0):
        # Switch x and y, and switch again when returning.
        xx, yy, val = weighted_line(c0, r0, c1, r1, w, rmin=rmin, rmax=rmax)
        return (yy, xx, val)

    # At this point we know that the distance in columns (x) is greater
    # than that in rows (y). Possibly one more switch if c0 > c1.
    if c0 > c1:
        return weighted_line(r1, c1, r0, c0, w, rmin=rmin, rmax=rmax, negative_order=True)

    # The following is now always < 1 in abs
    slope = (r1 - r0) / (c1 - c0)

    # Adjust weight by the slope
    w *= np.sqrt(1 + np.abs(slope)) / 2

    # We write y as a function of x, because the slope is always <= 1
    # (in absolute value)
    x = np.arange(c0, c1 + 1, dtype=float)
    y = x * slope + (c1 * r0 - c0 * r1) / (c1 - c0)

    # Now instead of 2 values for y, we have 2*np.ceil(w/2).
    # All values are 1 except the upmost and bottommost.
    thickness = np.ceil(w / 2)
    yy = (np.floor(y).reshape(-1, 1) + np.arange(-thickness - 1, thickness + 2).reshape(1, -1))
    xx = np.repeat(x, yy.shape[1])
    vals = trapez(yy, y.reshape(-1, 1), w).flatten()

    yy = yy.flatten()

    # Exclude useless parts and those outside of the interval
    # to avoid parts outside of the picture
    mask = np.logical_and.reduce((yy >= rmin, yy < rmax, vals > 0))

    if negative_order:
        reverse_yy = yy[::-1]
        reverse_xx = xx[::-1]
        reverse_vals = vals[::-1]
        return (reverse_yy[mask].astype(int), reverse_xx[mask].astype(int), reverse_vals[mask])
    return (yy[mask].astype(int), xx[mask].astype(int), vals[mask])


def draw_path(board_im, path):
    # choose length of the path until it goes red
    total_length = BOARD_BASE_HEIGHT * RESOLUTION_FACTOR   # 3 is the number of wall bounces before the path goes red
    # Initialize cumulative length
    cumulative_length = 0

    for i in range(len(path) - 1):
        # Calculate line coordinates and anti-aliasing weights
        rr, cc, val = weighted_line(path[i][1], path[i][0], path[i + 1][1], path[i + 1][0], 5)

        # Calculate length of the current segment
        segment_length = np.linalg.norm(np.array(path[i + 1]) - np.array(path[i]))

        # Calculate color gradient for each point on the line
        color_gradient = np.linspace(cumulative_length / total_length,
                                     (cumulative_length + segment_length) / total_length, len(rr))

        # Update cumulative length
        cumulative_length += segment_length

        if cumulative_length > total_length:
            return

        for r, c, v, cg in zip(rr, cc, val, color_gradient):
            if 0 <= r < board_im.shape[0] and 0 <= c < board_im.shape[1]:
                if cg < 0.5:
                    # Transition from green to yellow in the first half of the path
                    color = (1 - 2 * cg) * np.array(Colors.GREEN) + 2 * cg * np.array(Colors.AQUA)
                else:
                    # Transition from yellow to red in the second half of the path
                    color = (2 - 2 * cg) * np.array(Colors.AQUA) + (2 * cg - 1) * np.array(Colors.BLUE)
                board_im[r, c] = (1 - v) * board_im[r, c] + v * color
