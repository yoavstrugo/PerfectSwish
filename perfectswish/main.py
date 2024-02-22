import cv2
import numpy as np

from perfectswish.api import webcam
from perfectswish.api.live_image_display import LiveImageDisplay

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

    np.save(f"{DATA_DIRECTORY}/camera_rect.npy", np_camera_rect)
    np.save(f"{DATA_DIRECTORY}/projector_rect.npy", np_projector_rect)


def load_rects() -> tuple:
    """
    Load the camera and projector rects from a file.
    :return:
    """
    try:
        np_camera_rect = np.load(f"{DATA_DIRECTORY}/camera_rect.npy")
        np_projector_rect = np.load(f"{DATA_DIRECTORY}/projector_rect.npy")
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

    camera_rect = get_camera_rect(webcam.get_webcam_image(cap), initial_camera_rect)
    projector_rect = get_projector_rect()


def main_loop(cap: cv2.VideoCapture, camera_rect: list, projector_rect: list) -> np.array:
    webcam_image = webcam.get_webcam_image(cap)
    return webcam_image


def main():
    try:
        cap = webcam.initialize_webcam()
        camera_rect, projector_rect = setup_rects(cap)
    except Exception as e:
        print("Error setting up camera and projector")
        print(e)
        return

    liveImageDisplay = LiveImageDisplay(main_loop, cap, camera_rect, projector_rect,
                                        window_name="Perfect Swish", framerate=30)

    liveImageDisplay.run()


if __name__ == '__main__':
    main()
