import cv2
import tkinter as tk
from tkinter import filedialog

import numpy as np
from PIL import Image, ImageTk

from perfectswish.image_transformation.gui_api import get_rect
from perfectswish.image_transformation.image_processing import transform_board, generate_projection

from perfectswish.image_transformation import gui_class


class CameraRectApp(gui_class.CalibrationApp):

    def __init__(self, image, set_rect, rect=None):
        super().__init__(image, set_rect, rect)
        self.counter = 0
        # canvas contains the original image and the rectangle
        cv2.imshow("image", self.image)
        cv2.waitKey(0)
        self.canvas = tk.Canvas(self.root, width=int(self.image.shape[1] * self.scale_factor),
                                height=int(self.image.shape[0] * self.scale_factor))
        self.canvas.pack(side=tk.RIGHT, padx=20, pady=10, expand=True, fill=tk.BOTH)
        # Create a button for saving the image
        self.save_black_bg_button = tk.Button(self.root, text="Save Board Black bg", command=self.save_black_bg)
        self.save_black_bg_button.pack(side=tk.TOP, pady=10)
        self.transform_and_display()
        self.draw_rect()

    def _transformation_func(self, image, rect):
        return transform_board(image, rect)

    def draw_rect(self):
        # Draw the original image on the original canvas
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        scale_factor = self.scale_factor
        small_height = int(image_rgb.shape[0] * scale_factor)
        small_width = int(image_rgb.shape[1] * scale_factor)
        image_small = cv2.resize(image_rgb, (small_width, small_height))

        img_tk = ImageTk.PhotoImage(image=Image.fromarray(image_small))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk

        # Draw the rectangle on the original canvas
        self.canvas.create_polygon(self.rect[0], self.rect[1], self.rect[2], self.rect[3],
                                   self.rect[4], self.rect[5], self.rect[6], self.rect[7], outline="red",
                                   fill="")

        # Update the Tkinter window
        self.root.update()

        # Call the draw_rect function again after a delay (in milliseconds)
        self.root.after(100, self.draw_rect)

    def save_black_bg(self):
        # Transform the image using the specified rectangle
        actual_rect = [int(x / self.scale_factor) for x in self.rect]
        transformed_image = self._transformation_func(self.image, actual_rect)
        # Display the transformed image on the canvas
        image_with_black = generate_projection(transformed_image, actual_rect)
        cv2.imwrite(fr"black_bg{self.counter}.jpg", image_with_black)
        self.counter += 1


def get_camera_rect(image, initial_rect=None):
    return get_rect(image, CameraRectApp, initial_rect=initial_rect)


if __name__ == '__main__':
    path = r"C:\Users\TLP-299\PycharmProjects\PerfectSwish\perfectswish\image_transformation\images\blank_board.jpg"
    image = cv2.imread(path)

    def get_rect(rect):
        print(rect)


    app = CameraRectApp(image, get_rect)
    app.root.mainloop()
