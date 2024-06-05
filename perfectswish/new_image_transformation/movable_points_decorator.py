from typing import Callable

from perfectswish.new_image_transformation.frame_decorator import FrameDecorator


class MovablePoints(FrameDecorator):
    """
    This class will create a frame where the user can select N points on the image. The points will be
    movable, both with the mouse and with the keyboard.
    """

    # TODO: min imaeg height width for resize

    def __init__(self, frame, fine_movement: int = 1, shared_data_key: str = 'screen_2', edit_other_win_points: Callable = None):
        super().__init__(frame)
        self.__fine_movement = fine_movement
        self.__bind_events()
        self.__shared_data_key = shared_data_key
        self.__edit_other_win_points = edit_other_win_points

    def __bind_events(self):
        # Bind arrow keys to move the selected point
        self._app.bind("<Left>", lambda _: self.__move_selected_point(-1, 0), add=True)
        self._app.bind("<Right>", lambda _: self.__move_selected_point(1, 0), add=True)
        self._app.bind("<Up>", lambda _: self.__move_selected_point(0, -1), add=True)
        self._app.bind("<Down>", lambda _: self.__move_selected_point(0, 1), add=True)

        # Bind drag and drop
        self._canvas.bind("<B1-Motion>", self.__on_drag, add=True)

    def __update_shared_data(self):
        points_in_image = [self._get_image_point(x, y) for x, y in self._points]
        self._app.shared_data[self.__shared_data_key] = points_in_image
        if self.__edit_other_win_points:
            self.__edit_other_win_points(points_in_image)




    def __move_selected_point(self, dx, dy):
        if self._selected_point is not None:
            x, y = self._points[self._selected_point]
            x += dx * self.__fine_movement
            y += dy * self.__fine_movement
            self._points[self._selected_point] = (x, y)

            # bound the points to the canvas
            x, y = self._points[self._selected_point]
            self._points[self._selected_point] = (max(0, min(self._width, x)), max(0, min(self._height, y)))
        self.__update_shared_data()

    def __on_drag(self, event):
        if self._selected_point is not None:
            self._points[self._selected_point] = (event.x, event.y)

            # bound the points to the canvas
            x, y = self._points[self._selected_point]
            self._points[self._selected_point] = (max(0, min(self._width, x)), max(0, min(self._height, y)))
        self.__update_shared_data()





