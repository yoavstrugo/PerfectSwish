import tkinter as tk
from typing import TypeVar

import cv2
from PIL import Image
from screeninfo import Monitor, get_monitors

from perfectswish.new_image_transformation.base_image_frame import BaseImageFrame
from perfectswish.new_image_transformation.image_transform_frame_decorator import ImageTransform
from perfectswish.new_image_transformation.movable_points_decorator import MovablePoints
from perfectswish.new_image_transformation.point_selection_frame import PointSelectionFrame
from perfectswish.new_image_transformation.point_selection_frame_decorator import PointSelection
from perfectswish.new_image_transformation.points_frame_decorator import Points
from perfectswish.new_image_transformation.user_action_frame import UserActionFrame

class PerfectSwishApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Perfect Swish")
        self.geometry("800x600")

        self.__frames = []
        self.shared_data = dict()

        self.resizable(False, False)
        self.__create_frames()

    def __create_frames(self):
        user_action_frame = UserActionFrame(self, self)
        image_edit_frame = PointSelection(
            MovablePoints(
                Points(
                    BaseImageFrame(user_action_frame, self, get_image),
                )
            )
        )
        user_action_frame.set_frame(image_edit_frame)
        self.__frames.append(user_action_frame)

        self.__frames[-1].pack(fill="both", expand=True)
        # this_screen = get_root_screen(self)
        # other_screen = DisplayApp.get_display_screen(this_screen)
        #
        # self.__top_level = DisplayApp(other_screen)
        # reacting_image = ImageTransform(
        #     Points(
        #         BaseImageFrame(self.__top_level, self, get_image),
        #         __set_points=lambda: self.shared_data.get('rect', []),
        #     )
        # )
        # self.__top_level.set_frame(reacting_image)


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