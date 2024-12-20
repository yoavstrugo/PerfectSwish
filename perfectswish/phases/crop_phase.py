from typing import Callable

from perfectswish.utils.webcam import WebcamCapture
from perfectswish.new_image_transformation.base_image_frame import BaseImageFrame
from perfectswish.new_image_transformation.movable_points_decorator import MovablePoints
from perfectswish.new_image_transformation.phase import Phase
from perfectswish.new_image_transformation.point_selection_frame_decorator import PointSelection
from perfectswish.new_image_transformation.points_frame_decorator import Points
from perfectswish.new_image_transformation.user_action_frame import UserActionFrame


class CropPhase(Phase):
    """
    The crop phase. This will allow the user to crop the image.
    """
    def __init__(self, app: "PerfectSwishApp", saved_data, next_func: Callable, back_func: Callable,
                 cap: WebcamCapture):
        """
        Initializes the crop phase.
        :param app: the main app.
        :param saved_data: the saved data to load.
        :param next_func: the next function.
        :param back_func: the back function.
        :param cap:
        """
        super().__init__("crop")
        self._frame = UserActionFrame(app, app, next_btn_action=next_func, back_btn_action=back_func)

        self.crop_rect = None
        if saved_data:
            self.crop_rect = saved_data

        image_crop_gui = MovablePoints(
            PointSelection(
                Points(
                    BaseImageFrame(self._frame, app, cap.get_latest_image),
                    initial_points=self.crop_rect,
                    connected_points=True,
                )
            )
        )

        image_crop_gui._pass_points.append(self.set_crop_rect)

        self._frame.set_frame(image_crop_gui)
        self._frame.pack(fill="both", expand=True)

    def get_data(self):
        return self.crop_rect

    def destroy(self):
        self._frame.pack_forget()
        self._frame.destroy()

    def set_crop_rect(self, pts):
        self.crop_rect = pts