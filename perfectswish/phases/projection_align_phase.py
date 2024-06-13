from screeninfo import Monitor

from perfectswish.utils.webcam import WebcamCapture
from perfectswish.new_image_transformation.base_image_frame import BaseImageFrame
from perfectswish.new_image_transformation.display_app import DisplayApp
from perfectswish.new_image_transformation.image_transform_frame_decorator import ImageTransform
from perfectswish.new_image_transformation.movable_points_decorator import MovablePoints
from perfectswish.new_image_transformation.phase import Phase
from perfectswish.new_image_transformation.points_frame_decorator import Points
from perfectswish.new_image_transformation.user_action_frame import UserActionFrame


class ProjectAlignPhase(Phase):
    def __init__(self, app: "PerfectSwishApp", other_screen: Monitor, saved_data,
                 crop_pts, cap: WebcamCapture, next_func, back_func):
        super().__init__("project_align")
        self.__frame = UserActionFrame(app, app, next_btn_action=next_func,
                                       back_btn_action=back_func)
        self.__top_lvl = DisplayApp(other_screen)
        reacting_image = ImageTransform(
            Points(
                BaseImageFrame(self.__top_lvl, app, cap.get_latest_image,
                               width=other_screen.width,
                               height=other_screen.height),
                initial_points=saved_data,
                connected_points=True
            ),
            reference_points=crop_pts
        )
        self.__top_lvl.set_frame(reacting_image)

        align_gui = ImageTransform(
            MovablePoints(
                Points(
                    BaseImageFrame(self.__frame, app, cap.get_latest_image),
                    initial_points=saved_data
                )
            ),
            reference_points=crop_pts
        )
        self.__project_pts = None
        align_gui._pass_points.append(reacting_image._set_points)
        align_gui._pass_points.append(self.__set_projection_pts)
        self.__frame.set_frame(align_gui)
        self.__frame.pack(fill="both", expand=True)

    def __set_projection_pts(self, pts):
        self.__project_pts = pts

    def destroy(self):
        self.__frame.pack_forget()
        self.__top_lvl.withdraw()
        self.__top_lvl.destroy()
        self.__frame.destroy()

    def get_data(self):
        return self.__project_pts
