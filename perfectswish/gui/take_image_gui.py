import cv2
import tkinter as tk

import numpy as np

from perfectswish.api import webcam
from perfectswish.api.live_image_display import ControlledLiveImageDisplay


def __create_take_image_controls(root, return_values, cap, title):
    """
    Create the controls for taking an image.
    :param root: The root window.
    :param return_values: The return values.
    :param cap: The camera capture.
    :return:
    """

    def take_image():
        """
        Take an image and save it.
        :return:
        """
        image = webcam.get_webcam_image(cap)
        return_values["image"] = image
        root.destroy()

    title_widget = tk.Label(root, text=title)
    title_widget.pack(side=tk.TOP, padx=10, pady=10)

    button = tk.Button(root, text="Take Image", command=take_image)
    button.pack(side=tk.RIGHT, padx=10, pady=10)

    canvas = tk.Canvas(root, width=900, height=1080)
    canvas.pack(side=tk.LEFT, padx=10, pady=10)
    return canvas, [button]


def take_image(cap: cv2.VideoCapture, title: str) -> np.array:
    """
    Take an image and return it.
    :param cap: The camera capture.
    :return: The image.
    """

    imageDisplay = ControlledLiveImageDisplay(webcam.get_webcam_image, __create_take_image_controls, cap,
                                              title, window_name=title)
    imageDisplay.run()  # run until quit

    return imageDisplay.return_values["image"]
