from typing import Callable

from perfectswish.new_image_transformation.frame_decorator import FrameDecorator
from perfectswish.utils.trasnformation_pipeline import TransformationPipline


class Points(FrameDecorator):
    """
    This class will create a frame where the user can select N points on the image. The points will be
    static, should be used as a decorator to add data for other decorator depending on the points.
    """

    def __init__(self, frame, initial_points: list = None, points_in_relation_to: tuple = None,
                 render_points: bool = True, radius_tolerance: int = 10, connected_points: bool = False,
                 point_color: (int, int, int) = (0, 0, 255),
                 selected_point_color: (int, int, int) = (0, 255, 0),
                 point_radius: int = 3, point_padding: int = 2, pass_points: list[Callable] = None):
        """
        Initiate the Points frame decorator.
        :param frame: The frame to decorate.
        :param initial_points: Initial points to set on the canvas, in the coords of the image or in the
        coords of points_in_relation_to, if given.
        :param points_in_relation_to: Pass if the initial points are in the coords of another image.
        :param render_points: Whether to render the points on the canvas.
        :param radius_tolerance: Points will be selected if the user clicks on them within this radius.
        :param connected_points: Whether to connect the points with lines.
        :param point_color: The color of the points.
        :param selected_point_color: The color of the selected point.
        :param point_radius: The radius of the points.
        :param point_padding: The padding of the points from the edge of the canvas.
        :param pass_points: A list of functions that will be called with the points in the image coords.
        """
        # Initiate the points
        super().__init__(frame)
        self.__points_in_relation_to = points_in_relation_to
        self._set_points(initial_points or [], points_in_relation_to=points_in_relation_to)

        self._pass_points = pass_points or []

        self._transform_point: TransformationPipline[(int, int)] = TransformationPipline()
        self._selected_point = None
        self.__connected_points = connected_points
        self.__point_color = point_color
        self.__selected_point_color = selected_point_color
        self.__point_radius = point_radius
        self.__point_padding = point_padding
        self.__radius_tolerance = radius_tolerance
        if render_points:
            self._draw_after.append(self.__draw_points)

        self.__bind_events()

    def _set_points(self, points: list, points_in_relation_to: tuple = None) -> None:
        """
        This function will set the points, the list given is 'image' (or points_in_relation points, if given).
        :param points: Points in the coords of the image, or in the coords of points_in_relation_to, if given.
        """
        self._points = points
        self._max_points = len(self._points)
        if points_in_relation_to is None:
            self._points = [self._get_canvas_point(x, y) for x, y in self._points]
        else:
            self._points = [
                self.convert_point_coords(x, y, points_in_relation_to, (self._width, self._height))
                for x, y in self._points]

    def __bind_events(self):
        """
        This function will bind the events to the canvas.
        """
        self._canvas.bind("<Button-1>", self.__on_mouse_press, add=True)

    def __select_point(self, px, py):
        """
        This function will select a point if it's close enough to the given coords.
        :return:
        """
        for i, (x, y) in enumerate(self._points):
            if abs(px - x) < self.__radius_tolerance and abs(py - y) < self.__radius_tolerance:
                self._selected_point = i
                return i
        self._selected_point = None
        return None

    def __on_mouse_press(self, event):
        """
        This function will be called when the user clicks on the canvas. it will select the point if the user
        clicked on (next) a point.
        """
        self.__select_point(event.x, event.y)

    @staticmethod
    def convert_point_coords(x: int, y: int, src_coords: tuple, dst_coords: tuple) -> tuple:
        """
        Convert point x,y from a source coords to destination coords (proportionally, if point was sampled
        in context of 1920x1080, and you want to convert it to 1280x720, you should pass the source coords as
        (1920, 1080) and the destination coords as (1280, 720))
        """
        scale_x = dst_coords[0] / src_coords[0]
        scale_y = dst_coords[1] / src_coords[1]
        return x * scale_x, y * scale_y

    def _get_canvas_point(self, x: int, y: int) -> tuple:
        """
        This function will return the canvas point from the image point.
        """
        return self.convert_point_coords(x, y, (self._img_orig_width, self._img_orig_height),
                                         (self._width, self._height))

    def _get_image_point(self, x: int, y: int) -> tuple:
        """
        This function will return the image point from the canvas point.
        """
        return self.convert_point_coords(x, y, (self._width, self._height),
                                         (self._img_orig_width, self._img_orig_height))

    def __draw_points(self) -> None:
        """
        This function will draw the points on the canvas.
        """
        for pass_points_func in self._pass_points:
            image_points = [self._get_image_point(x, y) for x, y in self._points]
            pass_points_func(image_points)

        # draw the points through tkinter
        for i, (x, y) in enumerate(self._points):
            if i == self._selected_point:
                color = self.__selected_point_color
            else:
                color = self.__point_color

            # make sure the points are not out of the canvas
            if x - self.__point_radius < 0:
                x = max(x, self.__point_radius + self.__point_padding)
            elif x + self.__point_radius > self._width:
                x = min(x, self._width - self.__point_radius - self.__point_padding)

            if y - self.__point_radius < 0:
                y = max(y, self.__point_radius + self.__point_padding)
            elif y + self.__point_radius > self._height:
                y = min(y, self._height - self.__point_radius - self.__point_padding)

            # draw points
            self._canvas.create_oval(x - self.__point_radius, y - self.__point_radius,
                                     x + self.__point_radius, y + self.__point_radius,
                                     fill='#%02x%02x%02x' % color)

            # draw points text, some trail-and-error to make it look good
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

        # draw lines between points
        if self.__connected_points:
            for i in range(len(self._points) - 1, -1, -1):
                x1, y1 = self._points[i]
                x2, y2 = self._points[i - 1]
                self._canvas.create_line(x1, y1, x2, y2, fill='#%02x%02x%02x' % self.__point_color)

