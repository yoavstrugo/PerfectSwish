import cv2
import numpy as np

from perfectswish.api import webcam
from perfectswish.api.live_image_display import LiveImageDisplay


def setup_rects(cap: cv2.VideoCapture) -> tuple:
    """
    Setup the camera and projector rects.
    :param cap:
    :return:
    """
    return None, None


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
