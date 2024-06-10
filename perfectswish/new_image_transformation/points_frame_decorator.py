from typing import Callable

from perfectswish.new_image_transformation.frame_decorator import FrameDecorator
from perfectswish.utils.trasnformation_pipeline import TransformationPipline


class Points(FrameDecorator):
    """
    This class will create a frame where the user can select N points on the image. The points will be
    static, should be used as a decorator to add data for other decorator depending on the points.
    """

    # TODO: min imaeg height width for resize

    def __init__(self, frame, initial_points: list = None, render_points: bool = True,
                 radius_tolerance: int = 10, point_color: (int, int, int) = (0, 0, 255),
                 selected_point_color: (int, int, int) = (0, 255, 0),
                 point_radius: int = 3, point_padding: int = 2, pass_points: list[Callable] = None, **kwargs):
        # Initiate the points
        super().__init__(frame, **kwargs)
        self._set_points(initial_points or [])

        self._pass_points = pass_points or []

        self._transform_point: TransformationPipline[(int, int)] = TransformationPipline()
        self._selected_point = None
        # if render_points:
        #     self._transform_image.compose(self.__get_image_with_points, last=True)

        self.__point_color = point_color
        self.__selected_point_color = selected_point_color
        self.__point_radius = point_radius
        self.__point_padding = point_padding
        self.__radius_tolerance = radius_tolerance
        if render_points:
            self._draw_after.append(self.__draw_points)

        self.__bind_events()

    def _set_points(self, points: list):
        self._points = points
        self._max_points = len(self._points)
        self._points = [self._get_canvas_point(x, y) for x, y in self._points]

    def __bind_events(self):
        # Bind the click event
        self._canvas.bind("<Button-1>", self.__on_mouse_press, add=True)

    def __select_point(self, px, py):
        for i, (x, y) in enumerate(self._points):
            if abs(px - x) < self.__radius_tolerance and abs(py - y) < self.__radius_tolerance:
                self._selected_point = i
                return i
        self._selected_point = None
        return None

    def __on_mouse_press(self, event):
        self.__select_point(event.x, event.y)

    @staticmethod
    def convert_point_coords(x, y, src_coords, dst_coords) -> tuple:
        """
        Convert point x,y from a source coords to destination coords (proportionally, if point was sampled
        in context of 1920x1080, and you want to convert it to 1280x720, you should pass the source coords as
        (1920, 1080) and the destination coords as (1280, 720))
        """
        scale_x = dst_coords[0] / src_coords[0]
        scale_y = dst_coords[1] / src_coords[1]
        return x * scale_x, y * scale_y

    def _get_canvas_point(self, x, y):
        """
        This function will return the canvas point from the image point.
        """
        return self.convert_point_coords(x, y, (self._img_orig_width, self._img_orig_height), (self._width, self._height))

    def _get_image_point(self, x, y):
        """
        This function will return the image point from the canvas point.
        """
        return self.convert_point_coords(x, y, (self._width, self._height), (self._img_orig_width, self._img_orig_height))

    def __draw_points(self):
        for pass_points_func in self._pass_points:
            image_points = [self._get_image_point(x, y) for x, y in self._points]
            pass_points_func(image_points)

        # draw the points through tkinter
        for i, (x, y) in enumerate(self._points):
            if i == self._selected_point:
                color = self.__selected_point_color
            else:
                color = self.__point_color

            if x - self.__point_radius < 0:
                x = max(x, self.__point_radius + self.__point_padding)
            elif x + self.__point_radius > self._width:
                x = min(x, self._width - self.__point_radius - self.__point_padding)

            if y - self.__point_radius < 0:
                y = max(y, self.__point_radius + self.__point_padding)
            elif y + self.__point_radius > self._height:
                y = min(y, self._height - self.__point_radius - self.__point_padding)

            self._canvas.create_oval(x - self.__point_radius, y - self.__point_radius,
                                     x + self.__point_radius, y + self.__point_radius,
                                     fill='#%02x%02x%02x' % color)

            text_x, text_y = x + 5, y + 5
            if x <= self._width / 2 and y <= self._height / 2:
                text_x, text_y = x + 8, y + 8
            if x <= self._width / 2 and y > self._height / 2:
                text_x, text_y = x + 8, y - 8
            if x > self._width / 2 and y <= self._height / 2:
                text_x, text_y = x - 8, y + 8
            if x > self._width / 2 and y > self._height / 2:
                text_x, text_y = x - 8, y - 8

            self._canvas.create_text(text_x, text_y, text=str(i + 1), fill='#%02x%02x%02x' % color)