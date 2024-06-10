import tkinter as tk
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageTk

from perfectswish.utils.trasnformation_pipeline import TransformationPipline


class BaseImageFrame(tk.Frame):
    """
    This class will create a frame where the user can select N points on the image. The points will be
    movable, both with the mouse and with the keyboard.
    """

    # TODO: min imaeg height width for resize

    def __init__(self, master, app, get_image: Callable, fps: int = 24, width: int = 800, height: int =
    450):
        super().__init__(master)
        self._app = app
        self._master = master
        self._get_image = get_image
        self._width = width
        self._height = height
        self._img_orig_height, self._img_orig_width, _ = self._get_image().shape

        self._draw_before = []
        self._draw_after = []
        self._update = []
        self.__image_id = None

        self.__fps = fps

        self._transform_image: TransformationPipline[np.ndarray] = TransformationPipline()

        self.__create_widgets()
        self.__bind_events()
        self.__after_id = self.after(100, self.__update)
        self.__destroyed = False

    def destroy(self):
        self.after_cancel(self.__after_id)
        self.__destroyed = True


    def __bind_events(self):
        # Bind window resize
        self.bind("<Configure>", self.__on_resize)

    def __update(self):
        self.__draw()
        for to_update in self._update:
            to_update()

        if not self.__destroyed:
            self.__after_id = self.after(1000 // self.__fps, self.__update)



    def __resize_image(self, image):
        """
        This function will resize the image to fit the canvas.
        """
        return cv2.resize(image, (self._width, self._height), interpolation=cv2.INTER_LINEAR)

    def __draw(self):
        """
        This function will draw the image on the canvas.
        """
        self._canvas.delete("all")

        for draw in self._draw_before:
            draw()

        transformed_image = self.__resize_image(self._transform_image(self._get_image()))
        pil_image = Image.fromarray(transformed_image)
        tk_image = ImageTk.PhotoImage(pil_image)
        self._canvas.image = tk_image

        self.__image_id = self._canvas.create_image(0, 0, image=self._canvas.image, anchor="nw")

        for draw in self._draw_after:
            draw()



    def __on_resize(self, event):
        """
        When the frame is resized, the image will be resized as well, to fit the resized canvas
        """
        self.__draw()

    def __create_widgets(self):
        # Create a canvas to fit the entire frame
        self._canvas = tk.Canvas(self, width=self._width, height=self._height)
        self._canvas.pack(anchor="center")