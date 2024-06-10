from typing import Callable

from perfectswish.new_image_transformation.frame_decorator import FrameDecorator


class MovablePoints(FrameDecorator):
    """
    This class will create a frame where the user can select N points on the image. The points will be
    movable, both with the mouse and with the keyboard.
    """

    def __init__(self, frame, fine_movement: int = 1):
        """
        Initializes the MovablePoints frame.
        :param frame: The frame to decorate.
        :param fine_movement: How much pixels to move the selected point when using the arrow keys.
        """
        super().__init__(frame)
        self.__fine_movement = fine_movement
        self.__bind_events()

    def __bind_events(self):
        # Bind arrow keys to move the selected point
        self._app.bind("<Left>", lambda _: self.__move_selected_point(-1, 0), add=True)
        self._app.bind("<Right>", lambda _: self.__move_selected_point(1, 0), add=True)
        self._app.bind("<Up>", lambda _: self.__move_selected_point(0, -1), add=True)
        self._app.bind("<Down>", lambda _: self.__move_selected_point(0, 1), add=True)

        # Bind drag and drop
        self._canvas.bind("<B1-Motion>", self.__on_drag, add=True)

    def __move_selected_point(self, dx, dy) -> None:
        """
        Handles the fine movement of the selected point.
        :return:
        """
        if self._selected_point is not None:
            x, y = self._points[self._selected_point]
            x += dx * self.__fine_movement
            y += dy * self.__fine_movement
            self._points[self._selected_point] = (x, y)

            # bound the points to the canvas
            x, y = self._points[self._selected_point]
            self._points[self._selected_point] = (max(0, min(self._width, x)), max(0, min(self._height, y)))

    def __on_drag(self, event) -> None:
        """
        Handles the dragging of the selected point.
        """
        if self._selected_point is not None:
            self._points[self._selected_point] = (event.x, event.y)

            # bound the points to the canvas
            x, y = self._points[self._selected_point]
            self._points[self._selected_point] = (max(0, min(self._width, x)), max(0, min(self._height, y)))





