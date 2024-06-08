import threading
import time

import cv2
import numpy as np


class WebcamCapture:
    __EMPTY_IMAGE = np.zeros((1080, 1920, 3), dtype=np.uint8)
    def __init__(self, index: int, width: int, height: int, fps: int):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise IOError("Could not open video device")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))
        # set focus to 0
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap.set(cv2.CAP_PROP_FOCUS, 0)

        self.stop_flag = False
        self.capture_interval = 1 / fps
        self.lock = threading.Lock()
        self.latest_image = None
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()


    def _capture_loop(self):
        while not self.stop_flag:
            ret, frame = self.cap.read()
            with self.lock:
                self.latest_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            time.sleep(self.capture_interval)

    def get_latest_image(self):
        with self.lock:
            if self.latest_image is None:
                return self.__EMPTY_IMAGE
            return self.latest_image

    def release(self):
        self.stop_flag = True
        self.capture_thread.join()
        self.cap.release()


def initialize_webcam(index: int = 0, width=1920, height=1080, fps=30) -> cv2.VideoCapture:
    """
    Initializes the webcam.

    :param index: The index of the webcam to use.
    :param width: The width of the webcam video capture.
    :param height: The height of the webcam video capture.
    :param fps: The frames per second of the webcam video capture.

    :return: The webcam video capture.

    :raises IOError: If the webcam could not be opened.
    """
    # Initialize the webcam
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise IOError("Could not open video device")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))
    # set focus to 0
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, 0)
    return cap


def get_webcam_image(cap: cv2.VideoCapture) -> np.array:
    """
    Gets an image from the webcam.

    :param cap: The webcam video capture.
    :return: The image from the webcam.

    :raises IOError: If the image could not be read from the webcam.
    """
    ret, frame = cap.read()
    if not ret:
        raise IOError("Error reading from camera")

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
