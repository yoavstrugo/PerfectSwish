from typing import Callable

from perfectswish.new_image_transformation.frame_decorator import FrameDecorator


class PointSelection(FrameDecorator):
    def __init__(self, frame, n: int = 4, set_points_computer_frame_3: Callable = None, shared_data_key: str = ''):
        super().__init__(frame)
        self._max_points = n
        self.set_points_computer_frame_3 = set_points_computer_frame_3
        self.__bind_events()

    def __bind_events(self):
        # Bind delete key to delete the selected point
        self._app.bind("<Delete>", lambda _: self.__delete_selected_point())

        # Bind the click event
        self._canvas.bind("<Button-1>", self.__on_mouse_press, add=True)

    def __on_mouse_press(self, event):
        if len(self._points) < self._max_points:
            # convert the click coordinates to the image coordinates, which is the canvas coordinates
            self._points.append((event.x, event.y))
            self._selected_point = len(self._points) - 1
            self.__update_frame_3_computer()

    def __delete_selected_point(self):
        if self._selected_point is not None:
            del self._points[self._selected_point]
            self._selected_point = None
            self.__update_frame_3_computer()
