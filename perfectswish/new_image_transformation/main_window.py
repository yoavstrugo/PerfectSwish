import tkinter as tk

import cv2
import numpy as np
from PIL import Image
from screeninfo import Monitor, get_monitors

from perfectswish.api import webcam
from perfectswish.image_transformation.image_processing import Board, transform_board
from perfectswish.new_image_transformation.base_image_frame import BaseImageFrame
from perfectswish.new_image_transformation.image_transform_frame_decorator import ImageTransform
from perfectswish.new_image_transformation.movable_points_decorator import MovablePoints
from perfectswish.new_image_transformation.point_selection_frame_decorator import PointSelection
from perfectswish.new_image_transformation.points_frame_decorator import Points
from perfectswish.new_image_transformation.user_action_frame import UserActionFrame
from perfectswish.object_detection.detect_balls import draw_circles, find_balls
from perfectswish.object_detection.detect_cuestick import CuestickDetector
from perfectswish.visualization.visualize import draw_board

CAMERA = 1


cap = webcam.initialize_webcam(CAMERA)
cuestick_detector = CuestickDetector()
RESOLUTION_FACTOR = 8

class SharedDataEntry:
    def __init__(self, data):
        self.data = data

    def set(self, data):
        self.data = data

    def get(self):
        return self.data

class PerfectSwishApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Perfect Swish")
        self.geometry("800x600")

        self.__frames: list[tuple[tk.Frame, tk.Toplevel | None]] = []
        self.shared_data = dict()

        self.crop_rect = None
        self.resizable(False, False)
        self.__create_frames()
        self.set_frame(0)

    def set_frame(self, frame_num):
        # close all the windows
        for frame in self.__frames:
            if frame[1]:
                frame[1].withdraw()
                frame[1].show(False)
            # hide the current window
            frame[0].pack_forget()
            frame[0].show(False)
        # show the selected window
        self.__frames[frame_num][0].pack(fill="both", expand=True)
        self.__frames[frame_num][0].show(True)
        if self.__frames[frame_num][1]:
            self.__frames[frame_num][1].deiconify()
            self.__frames[frame_num][1].show(True)


    def set_crop_rect(self, pts):
        self.crop_rect = pts

    def main_loop(self) -> (
            np.array):
        webcam_image = webcam.get_webcam_image(cap)
        if not self.crop_rect:
            return webcam_image
        cropped_board = transform_board(webcam_image, self.crop_rect)
        cv2.imwrite("board_cropped.png", cropped_board)

        balls = find_balls(cropped_board)
        cue = cuestick_detector.detect_cuestick(cropped_board)

        board_im = draw_board(Board(112 * RESOLUTION_FACTOR, 224 * RESOLUTION_FACTOR))
        if balls is not None:
            draw_circles(board_im, balls)

        if cue is not None:
            stickend, back_fiducial_center, front_fiducial_center = cue
            cuestick_detector.draw_cuestick(board_im, stickend, back_fiducial_center, front_fiducial_center)

        return board_im


    def __create_frames(self):
        first_frame = UserActionFrame(self, self, next_btn_action=lambda: self.set_frame(1),
                                      back_btn_action=None)
        image_frame1 = PointSelection(
            Points(
                BaseImageFrame(first_frame, self, lambda: webcam.get_webcam_image(cap)),
            ),
        )
        image_frame1._pass_points.append(self.set_crop_rect)
        first_frame.set_frame(image_frame1)
        self.__frames.append((first_frame, None))

        this_screen = get_root_screen(self)
        other_screen = DisplayApp.get_display_screen(this_screen)

        # frame 2
        frame2 = UserActionFrame(self, self, next_btn_action=lambda: self.set_frame(2),
                                 back_btn_action=lambda: self.set_frame(1))
        toplevel2 = DisplayApp(other_screen)
        reacting_image = ImageTransform(
            Points(
                BaseImageFrame(toplevel2, self, lambda: webcam.get_webcam_image(cap), width=other_screen.width,
                               height=other_screen.height)
            )
        )
        toplevel2.set_frame(reacting_image)
        image_frame2 = ImageTransform(MovablePoints(
            Points(
                BaseImageFrame(frame2, self, lambda: webcam.get_webcam_image(cap)),
            ), edit_other_win_points=reacting_image._set_points
        )
        )
        frame2.set_frame(image_frame2)
        self.__frames.append((frame2, toplevel2))
        image_frame1._pass_points.append(image_frame2._set_reference_points)
        image_frame1._pass_points.append(reacting_image._set_reference_points)

        # frame 3

        frame3 = UserActionFrame(self, self, next_btn_action=None,
                                 back_btn_action=lambda: self.set_frame(2))
        image_frame3 = ImageTransform(
            Points(
                BaseImageFrame(frame3, self, lambda: webcam.get_webcam_image(cap)),
            )

        )
        toplevel3 = DisplayApp(other_screen)
        image_frame3_projection = ImageTransform(Points(
            BaseImageFrame(toplevel3, self, self.main_loop, width=other_screen.width,
                           height=other_screen.height)))
        toplevel3.set_frame(image_frame3_projection)
        frame3.set_frame(image_frame3)
        self.__frames.append((frame3, toplevel3))

        image_frame1._pass_points.append(image_frame3._set_reference_points)
        image_frame1._pass_points.append(image_frame3_projection._set_reference_points)
        image_frame2._pass_points.append(image_frame3_projection._set_points)

class DisplayApp(tk.Toplevel):
    def __init__(self, screen: Monitor):
        super().__init__()
        self.__frame = None
        self.title("Display Image")

        # this sets an effective fullscreen
        self.geometry(f"{screen.width}x{screen.height}+{screen.x}+{screen.y}")
        self.resizable(False, False)
        self.attributes('', True)
        self.overrideredirect(True)
        self.attributes('-topmost', True)

    def show(self, show: bool):
        self.__frame.switch_show(show)

    def set_frame(self, frame):
        self.__frame = frame
        self.__frame.pack_forget()
        self.__frame.pack(fill="both", expand=True)

    @staticmethod
    def get_display_screen(control_screen):
        for monitor in get_monitors():
            if monitor != control_screen:
                return monitor
        return None


img = cv2.imread("chess.jpeg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_pil = Image.fromarray(img_rgb)


def get_image():
    return img_rgb


def get_root_screen(root: tk.Tk):
    for monitor in get_monitors():
        if monitor.x <= root.winfo_x() <= monitor.x + monitor.width and \
                monitor.y <= root.winfo_y() <= monitor.y + monitor.height:
            return monitor
    return None


class TestClass:
    def __set__(self, instance, value):
        pass


if __name__ == "__main__":
    app = PerfectSwishApp()
    app.mainloop()
