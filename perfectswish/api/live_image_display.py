import tkinter as tk

import cv2
from PIL import Image, ImageTk

FPS = 30
WIDTH = 1920
HEIGHT = 1080


class LiveImageDisplay:
    def __init__(self, main_loop, *args, framerate=FPS, window_name: str = '', width=WIDTH, height=HEIGHT):
        self.main_loop = main_loop
        self.args = args

        self.delay = 1000 // framerate
        self.width = width
        self.height = height

        self.images = []

        self.root = tk.Tk()
        self.root.title(window_name)

        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height)
        self.canvas.pack()

    def run(self):
        images = self.main_loop(*self.args)
        self.display_image(images)
        self.root.after(self.delay, self.run)
        self.root.update()

        self.root.mainloop()

    def display_image(self, images):
        self.images = []
        # check if images is iterable and display all images
        # if images is not None and hasattr(images, '__iter__'):
        #     # display multiple image
        #     image_width = self.width // len(images)
        #     for i, image in enumerate(images):
        #         image_pos = i * image_width
        #         image_resized = cv2.resize(image, (image_width, self.height), interpolation=cv2.INTER_AREA)
        #         image_tk = ImageTk.PhotoImage(image=Image.fromarray(image_resized))
        #         self.canvas.create_image(image_pos, 0, anchor=tk.NW, image=image_tk)
        #
        #         self.images.append(image_tk)
        #     return

        # display single image
        image_resized = cv2.resize(images, (self.width, self.height))
        image_tk = ImageTk.PhotoImage(image=Image.fromarray(image_resized))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
        self.canvas.image = image_tk
