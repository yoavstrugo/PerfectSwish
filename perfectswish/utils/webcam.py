import multiprocessing
import multiprocessing.shared_memory as shm
import threading
import time

import cv2
import numpy as np


class MultiprocessWebcamCapture:
    """
    This class is used to pass a WebcamCapture instance to a process.
    Should be created using the create_multiprocess method of WebcamCapture.
    """

    def __init__(self, shared_mem, lock, height, width):
        """
        Initializes the MultiprocessWebcamCapture.
        :param shared_mem:
        :param lock:
        :param height:
        :param width:
        """
        self.shared_mem = shared_mem
        self.lock = lock
        self.width = width
        self.height = height

    def get_latest_image(self) -> np.ndarray:
        """
        Gets the latest image from the webcam.
        """
        with self.lock:
            img = np.ndarray((self.width, self.height, 3), dtype=np.uint8,
                             buffer=self.shared_mem.buf)
            img.flags.writeable = False
            return img


class WebcamCapture:
    """
    This class captures video from a webcam and allows the user to get the latest image from the webcam.
    This class is concurrency safe - both for threads and processes.
    To pass an instance of this class to a process, use the create_multiprocess method.
    """

    def __init__(self, index: int | str, width: int, height: int, fps: int):
        """
        Initializes the webcam capture.
        :param index: The index of the webcam to use. You may also pass a string to use a video file instead.
        :param width: The width of the video capture.
        :param height: The height of the video capture.
        :param fps: The frames per second of the video capture.
        """
        self._setup_feed(index, width, height, fps)
        if not self.cap.isOpened():
            raise IOError("Could not open video device")
        self.width = width
        self.height = height
        self.__EMPTY_IMAGE = np.zeros((height, width, 3), dtype=np.uint8)

        self.shared_memory = shm.SharedMemory(create=True, size=height * width * 3)
        self.shared_img = np.ndarray((height, width, 3), dtype=np.uint8, buffer=self.shared_memory.buf)
        self.shared_img.flags.writeable = True
        self.shared_img[:] = self.__EMPTY_IMAGE

        self.stop_flag = False
        self.capture_interval = 1 / fps
        self.lock = multiprocessing.Lock()
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def _setup_feed(self, index, width, height, fps) -> None:
        """
        Sets up the video feed.
        """
        if isinstance(index, str):
            self.cap = cv2.VideoCapture(index)
        else:
            self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            # set focus to 0
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.cap.set(cv2.CAP_PROP_FOCUS, 0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))

    def create_multiprocess(self) -> MultiprocessWebcamCapture:
        """
        Creates a MultiprocessWebcamCapture instance that can be passed to a process.
        """
        return MultiprocessWebcamCapture(self.shared_memory, self.lock, self.width, self.height)

    def _capture_loop(self) -> None:
        """
        The loop that captures the video feed.
        """
        while not self.stop_flag:
            ret, frame = self.cap.read()
            with self.lock:
                self.shared_img.flags.writeable = True
                self.shared_img[:] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            time.sleep(self.capture_interval)

    def get_latest_image(self) -> np.ndarray:
        """
        Gets the latest image from the webcam.
        """
        with self.lock:
            self.shared_img.flags.writeable = False
            return self.shared_img

    def release(self) -> None:
        """
        Releases the webcam capture.
        """
        self.stop_flag = True
        self.capture_thread.join()
        self.cap.release()
        self.shared_memory.close()
        self.shared_memory.unlink()


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
