import tkinter as tk

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
                 display_last_image: bool = False,
                 width=WIDTH, height=HEIGHT):
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

        self.__image = None
        self.__blank_image = np.zeros((self.__height, self.__width, 3), dtype=np.uint8)

        self.__root = tk.Tk()
        self.__root.title(window_name)

        self.__canvas = tk.Canvas(self.__root, width=self.__width, height=self.__height)
        self.__canvas.pack()

    def __check_image(self, image):
        """
        Check if the image is a valid np.ndarray.
        :param image: The image to check.
        :return: True if the image is invalid, False otherwise.
        """
        if not isinstance(image, np.ndarray):
            print(f"Error: Invalid image type: {type(image)}")
            return True
        elif image.shape[:2] != (self.__height, self.__width):
            print(f"Error: Invalid image shape: {image.shape}")
            return True
        return False

    def __get_image_with_err(self):
        image_error = False
        try:
            image = self.__main_loop(*self.__args)
        except Exception as e:
            print(f"Error: In main loop: {e}")
            if self.__display_last_image:
                image = self.__image
            else:
                image = None
                image_error = True

        return image, self.__check_image(image) or image_error

    def run(self):
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

        self.__root.after(self.__delay, self.run)
        self.__root.update()
        self.__root.mainloop()

    def __display_image(self, images):
        # display single image
        image_resized = cv2.resize(images, (self.__width, self.__height))
        image_tk = ImageTk.PhotoImage(image=Image.fromarray(image_resized))
        self.__canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
        self.__image = image_tk
