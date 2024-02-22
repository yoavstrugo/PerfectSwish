import cv2
import numpy as np


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
    # set buffer
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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

    return frame
