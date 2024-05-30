from typing import Callable

from perfectswish.new_image_transformation.movable_points_frame import MovablePointsFrame


class PointSelectionFrame(MovablePointsFrame):
    """
    This class will create a frame where the user can select N points on the image. The points will be
    movable, both with the mouse and with the keyboard.
    """

    def __init__(self, master, app, get_image: Callable, n: int, initial_points: list = None,
                 radius_tolerance: int = 10, fps: int = 24, fine_movement: int = 1):
        super().__init__(master, app, get_image, initial_points, radius_tolerance, fps, fine_movement)

        # Initiate the points
        self._points = list()
        if initial_points is not None:
            self._points = [(0, 0) for _ in range(n)]

        self._max_points = n

    def _bind_events(self):
        # Bind delete key to delete the selected point
        self._app.bind("<Delete>", lambda _: self.__delete_selected_point())

    def _on_mouse_press(self, event):
        # Otherwise, add a new point, if there is space
        # TODO: tranform the click to real size image coords
        if len(self._points) < self._max_points:
            # convert the click coordinates to the image coordinates, which is the canvas coordinates
            self._points.append((event.x, event.y))
            self._selected_point = len(self._points) - 1

    def __delete_selected_point(self):
        if self._selected_point is not None:
            del self._points[self._selected_point]
            self._selected_point = None
