import tkinter as tk

import cv2
from PIL import Image
from screeninfo import Monitor, get_monitors

from perfectswish.new_image_transformation.base_image_frame import BaseImageFrame
from perfectswish.new_image_transformation.image_transform_frame_decorator import ImageTransform
from perfectswish.new_image_transformation.movable_points_decorator import MovablePoints
from perfectswish.new_image_transformation.points_frame_decorator import Points
from perfectswish.new_image_transformation.user_action_frame import UserActionFrame

CAMERA = 0


class PerfectSwishApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Perfect Swish")
        self.geometry("800x600")

        self.__frames: list[tuple[tk.Frame, tk.Toplevel | None]] = []
        self.shared_data = dict()

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

    def get_camera_image(self):
        # connect to camera and get image
        camera = cv2.VideoCapture(CAMERA)
        ret, frame = camera.read()
        camera.release()
        # convert to RGB
        return frame

    def __create_frames(self):
        camera_image_bgr = self.get_camera_image()
        camrea_image = lambda: cv2.cvtColor(camera_image_bgr, cv2.COLOR_BGR2RGB)
        chess_image = lambda: cv2.imread("chess.jpeg")
        first_frame = UserActionFrame(self, self, next_btn_action=lambda: self.set_frame(1), back_btn_action=None)
        image_frame1 = ImageTransform(MovablePoints(
            Points(
                BaseImageFrame(first_frame, self, camrea_image),
            )
        )
        )
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
                BaseImageFrame(toplevel2, self, camrea_image, width=other_screen.width,
                               height=other_screen.height),
            )
        )
        toplevel2.set_frame(reacting_image)
        image_edit_frame = ImageTransform(MovablePoints(
            Points(
                BaseImageFrame(frame2, self, camrea_image),
            ), edit_other_win_points=reacting_image._set_points
        )
        )
        frame2.set_frame(image_edit_frame)
        self.__frames.append((frame2, toplevel2))


        # frame 3
        frame3 = UserActionFrame(self, self, next_btn_action=None,
                                              back_btn_action=lambda: self.set_frame(2))
        image_frame3 = ImageTransform(Points(BaseImageFrame(frame3, self, camrea_image)))
        toplevel3 = DisplayApp(other_screen)
        image_frame3_projection = ImageTransform(Points(BaseImageFrame(toplevel3, self, camrea_image, width=other_screen.width, height=other_screen.height)))
        toplevel3.set_frame(image_frame3_projection)
        frame3.set_frame(image_frame3)
        self.__frames.append((frame3, toplevel3))



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
