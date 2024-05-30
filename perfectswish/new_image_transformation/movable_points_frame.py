import tkinter as tk
from typing import Callable, final

import cv2
import numpy as np
from PIL import Image, ImageTk

from perfectswish.new_image_transformation.trasnformation_pipeline import TransformationPipline


class MovablePointsFrame(tk.Frame):
    """
    This class will create a frame where the user can select N points on the image. The points will be
    movable, both with the mouse and with the keyboard.
    """

    # TODO: min imaeg height width for resize

    def __init__(self, master, app, get_image: Callable, points: list = None,
                 radius_tolerance: int = 10, fps: int = 24, fine_movement: int = 1,
                 image_ratio: (int, int) = (16, 9), width: int = 800, height: int = 450):
        super().__init__(master)
        self._app = app
        self._master = master
        self._get_image = get_image

        self._rendered = False

        self._width = width
        self._height = height

        # Initiate the points
        self._points = points or []

        self._max_points = len(self._points)
        self.__radius_tolerance = radius_tolerance
        self._selected_point = None
        self.__fps = fps
        self.__fine_movement = fine_movement

        self._transform_image: TransformationPipline[np.ndarray] = TransformationPipline()
        self._transform_point: TransformationPipline[(int, int)] = TransformationPipline()

        self.__create_widgets()
        self.__bind_events()
        self.after(100, self._on_canvas_render)
        self.after(100, self.__update)


    def _on_canvas_render(self):
        self._rendered = True

    def __bind_events(self):
        # Bind window resize
        self.bind("<Configure>", self.__on_resize)

        # Bind the click event
        self._canvas.bind("<Button-1>", self.__on_mouse_press)

        # Bind arrow keys to move the selected point
        self._app.bind("<Left>", lambda _: self.__move_selected_point(-1, 0))
        self._app.bind("<Right>", lambda _: self.__move_selected_point(1, 0))
        self._app.bind("<Up>", lambda _: self.__move_selected_point(0, -1))
        self._app.bind("<Down>", lambda _: self.__move_selected_point(0, 1))

        # Bind drag and drop
        self._canvas.bind("<B1-Motion>", self.__on_drag)


    def __move_selected_point(self, dx, dy):
        if self._selected_point is not None:
            x, y = self._points[self._selected_point]
            x += dx * self.__fine_movement
            y += dy * self.__fine_movement
            self._points[self._selected_point] = (x, y)

            # bound the points to the canvas
            x, y = self._points[self._selected_point]
            self._points[self._selected_point] = (max(0, min(self._width, x)), max(0, min(self._height, y)))

    def __select_point(self, px, py):
        for i, (x, y) in enumerate(self._points):
            if abs(px - x) < self.__radius_tolerance and abs(py - y) < self.__radius_tolerance:
                self._selected_point = i
                return i
        self._selected_point = None
        return None

    def __on_drag(self, event):
        if self._selected_point is not None:
            self._points[self._selected_point] = (event.x, event.y)

            # bound the points to the canvas
            x, y = self._points[self._selected_point]
            self._points[self._selected_point] = (max(0, min(self._width, x)), max(0, min(self._height, y)))

    def __on_mouse_press(self, event):
        """
        Handles the point selection on mouse press.
        """
        # Check if it's close to any of the points
        if self.__select_point(event.x, event.y) is not None:
            return

    def __get_image_with_points(self):
        """
        This function will return the image with the points drawn on it.
        """
        image = self._get_image()
        resized_image = self._transform_image(cv2.resize(image, (self._width, self._height)))
        transformed_image = resized_image
        point_radius = 3
        point_padding = 2
        for i, (x, y) in enumerate(self._points):
            if i == self._selected_point:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            # TODO: transform the points to the rescaled image
            # draw so everything wll be on the image (0..width, 0..height)
            if x - point_radius < 0:
                x = max(x, point_radius + point_padding)
            elif x + point_radius > self._width:
                x = min(x, self._width - point_radius - point_padding)

            if y - point_radius < 0:
                y = max(y, point_radius + point_padding)
            elif y + point_radius > self._height:
                y = min(y, self._height - point_radius - point_padding)

            # Draw the point
            cv2.circle(transformed_image, (x, y), 3, color, -1)

            # Select text position based on the point position
            # top-left, top-right, bottom-left, bottom-right
            text_x, text_y = x + 5, y + 5
            if x <= self._width / 2 and y <= self._height / 2:
                text_x, text_y = x + 5, y + 14
            if x <= self._width / 2 and y > self._height / 2:
                text_x, text_y = x + 6, y - 3
            if x > self._width / 2 and y <= self._height / 2:
                text_x, text_y = x - 15, y + 14
            if x > self._width / 2 and y > self._height / 2:
                text_x, text_y = x - 15, y - 3

            # Draw the number of the point
            cv2.putText(transformed_image, str(i + 1), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color,
                        1)
        return resized_image

    def __update(self):
        self.__draw()
        self.after(1000 // self.__fps, self.__update)

    def __draw(self):
        if not self._rendered:
            return
        self._canvas.delete("all")
        # Draw points on the image
        image_with_points = self.__get_image_with_points()
        transformed_image = image_with_points
        pil_image = Image.fromarray(transformed_image)
        tk_image = ImageTk.PhotoImage(pil_image)
        self._canvas.image = tk_image
        self._canvas.create_image(0, 0, image=self._canvas.image, anchor="nw")

    def __on_resize(self, event):
        """
        When the frame is resized, the image will be resized as well, to fit the resized canvas
        """
        self.__draw()

    def __create_widgets(self):
        # Create a canvas to fit the entire frame
        self._canvas = tk.Canvas(self, width=self._width, height=self._height)
        self._canvas.pack(anchor="center")