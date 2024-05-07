import tkinter as tk
import traceback

import cv2
import numpy as np
from PIL import Image, ImageTk

FPS = 30
WIDTH = 1920
HEIGHT = 1080


class LiveImageDisplay:
    """
    Continuously displays images from a main loop, to make a "live feed".
    """

    def __init__(self, main_loop, *args, framerate=FPS, window_name: str = '',
                 display_last_image: bool = False, borderless: bool = False,
                 width=WIDTH, height=HEIGHT, display_on_second_monitor: bool = False, fullscreen=True, remember_balls_func=None):
        """
        A class which continuously displays images from a main loop.
        :param main_loop: A function which returns an image.
        :param args: Arguments to pass to the main loop.
        :param framerate: The framerate to display the images at.
        :param window_name: The name of the window.
        :param display_last_image: Whether to display the previous image if the main loop returns None,
        invalid type, or raises an exception. If False, the blank image will be displayed.
        :param width: The width of the window.
        :param height: The height of the window.
        """
        self.__main_loop = main_loop
        self.__args = args

        self.__display_last_image = display_last_image

        self.__delay = 1000 // framerate
        self.__width = width
        self.__height = height

        self.__main_loop_arg_count = main_loop.__code__.co_argcount

        self.__image = None
        self.__blank_image = np.zeros((self.__height, self.__width, 3), dtype=np.uint8)

        self._root = tk.Tk()
        self._root.title(window_name)
        if borderless:
            self._root.overrideredirect(True)
        if display_on_second_monitor:
            second_screen_height = self._root.winfo_screenheight()
            second_screen_width = self._root.winfo_screenwidth()

            # Set initial position for the second screen (adjust as needed)
            second_screen_x = second_screen_width  # X-coordinate for the second screen
            second_screen_y = 0  # Y-coordinate for the second screen
            self._root.geometry(f"{second_screen_width}x{second_screen_height}+{second_screen_x}+{second_screen_y}")

        if fullscreen:
            self._canvas = tk.Canvas(self._root, width=self._root.winfo_screenwidth(), height= self._root.winfo_screenheight())
            self._canvas.pack(fill=tk.BOTH, expand=tk.YES)
        else:
            self._canvas = tk.Canvas(self._root, width=self.__width, height=self.__height)
            self._canvas.pack()

        self.remember_balls_func = remember_balls_func

    def __check_image(self, image):
        """
        Check if the image is a valid np.ndarray.
        :param image: The image to check.
        :return: True if the image is invalid, False otherwise.
        """
        if not isinstance(image, np.ndarray):
            print(f"Error: Invalid image type: {type(image)}")
            return True
        return False

    def __get_image_with_err(self):
        image_error = False
        try:
            image = self.__main_loop(*(self.__args[:self.__main_loop_arg_count]))
        except Exception as e:
            print(f"Error: In main loop: {traceback.format_exc()}")
            if self.__display_last_image:
                image = self.__image
            else:
                image = None
                image_error = True

        return image, self.__check_image(image) or image_error

    def run(self):
        self.__run()
        self._root.mainloop()

    def __run(self):
        """
        Run the main loop and display the images.
        """
        image, err = self.__get_image_with_err()

        if err:
            # Display blank if there is an error
            if not self.__display_last_image:
                self.__display_image(self.__blank_image)
        else:
            self.__display_image(image)

        self._root.after(self.__delay, self.__run)

    def __display_image(self, images):
        # display single image
        image_resized = cv2.resize(images, (self._root.winfo_screenwidth(), self._root.winfo_screenheight()))
        image_tk = ImageTk.PhotoImage(image=Image.fromarray(image_resized))
        self._canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
        self.__image = image_tk


class ControlledLiveImageDisplay(LiveImageDisplay):
    """
    A live image display with control buttons.
    """

    def __init__(self, main_loop, create_controls, *args, framerate=FPS, window_name: str = '',
                 display_last_image: bool = False,
                 width=WIDTH, height=HEIGHT):
        """
        A class which continuously displays images from a main loop.
        :param main_loop: A function which returns an image.
        :param args: Arguments to pass to the main
        :param framerate: The framerate to display the images at.
        :param window_name: The name of the window.
        :param display_last_image: Whether to display the previous image if the main loop returns None,
        invalid type, or raises an exception. If False, the blank image will be displayed.
        :param width: The width of the window.
        :param height: The height of the window.
        """

        width = 900
        height = 1080

        super().__init__(main_loop, *args, framerate=framerate, window_name=window_name,
                         display_last_image=display_last_image, width=width, height=height)

        self.return_values = dict()

        self._canvas.pack_forget()
        self._canvas, self.__controls = create_controls(self._root, self.return_values, *args)
