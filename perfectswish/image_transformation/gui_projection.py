import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk

from perfectswish.image_transformation import gui_class
from perfectswish.image_transformation.image_processing import generate_projection
from perfectswish.image_transformation.gui_api import get_rect


class ProjectionRectApp(gui_class.CalibrationApp):
    def __init__(self, image, set_rect, rect=None):
        super().__init__(image, set_rect, rect, scale_factor=0.4)
        self.root.title("Projection Rectangle Adjustment")
        self.second_screen_display = tk.Toplevel(self.root)
        self.initialize_cropped_image_window()
        self.draw_rect()
    def initialize_cropped_image_window(self):
        # self.canvas_transformed.attributes('-topmost', True)  # Bring to front
        # self.canvas_transformed.attributes('-alpha', 0.7)  # Set transparency (adjust as needed)

        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Set initial position for the second screen (adjust as needed)
        second_screen_x = screen_width  # X-coordinate for the second screen
        second_screen_y = 0  # Y-coordinate for the second screen

        # Set the initial position of the window
        self.second_screen_display.geometry(f"{screen_width}x{screen_height}+{second_screen_x}+{second_screen_y}")

        # Set the window attributes
        self.second_screen_display.overrideredirect(True)

        # Create a canvas for displaying the transformed image
        self.projected_image_canvas = tk.Canvas(self.second_screen_display, width=screen_width, height=screen_height)
        self.projected_image_canvas.pack(fill=tk.BOTH, expand=tk.YES)

    def draw_rect(self):
        # Draw the original image on the original canvas
        self.transform_and_display()
        # Update the Tkinter window
        self.root.update()

        # draw the cropped image on the second screen and on the canvas
        if self.cropped_image is not None:
            cropped_image_rgb = cv2.cvtColor(self.cropped_image, cv2.COLOR_BGR2RGB)
            small_height = self.root.winfo_screenheight()
            small_width = self.root.winfo_screenwidth()
            image_small = cv2.resize(cropped_image_rgb, (small_width, small_height))
            img_tk = ImageTk.PhotoImage(image=Image.fromarray(image_small))
            self.projected_image_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.projected_image_canvas.image = img_tk

        # Call the draw_rect function again after a delay (in milliseconds)
        self.root.after(50, self.draw_rect)
    def _transformation_func(self, image, rect):
        return generate_projection(image, rect)

def get_projection_rect(image, initial_rect=None):
    return get_rect(image, ProjectionRectApp, initial_rect=initial_rect)


if __name__ == '__main__':
    image_path = r"C:\Users\TLP-299\PycharmProjects\computer-vision-pool\uncropped_images\board1_uncropped.jpg"
    initial_rect = np.array([820, 320, 1100, 300, 1400, 850, 709, 831])  # Initial rectangle coordinates
    image_in = cv2.imread(image_path)
    rect = get_projection_rect(image_in, initial_rect=initial_rect)
    print(rect)
    cv2.imshow("Hi!", generate_projection(image_in, rect))
    cv2.waitKey(0)
