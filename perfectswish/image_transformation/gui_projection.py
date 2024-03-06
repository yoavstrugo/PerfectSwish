import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog

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
        screen_width = 1920 # self.root.winfo_screenwidth()
        screen_height = 1080 # self.root.winfo_screenheight()

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
            scale_factor = self.scale_factor
            small_height = int(cropped_image_rgb.shape[0] * scale_factor)
            small_width = int(cropped_image_rgb.shape[1] * scale_factor)
            image_small = cv2.resize(cropped_image_rgb, (small_width, small_height))
            img_tk = ImageTk.PhotoImage(image=Image.fromarray(cropped_image_rgb))
            self.projected_image_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.projected_image_canvas.image = img_tk

        # Call the draw_rect function again after a delay (in milliseconds)
        self.root.after(50, self.draw_rect)
    def _transformation_func(self, image, rect):
        return generate_projection(image, rect)

class RectAdjustmentAppProjection:
    def __init__(self, image, set_rect, rect=None, real_image=True):

        if rect is None:
            rect = [int(0.4 * x) for x in [817, 324, 1186, 329, 1364, 836, 709, 831]]

        self.image = image
        if not real_image:
            self.image = np.ones_like(image) * 200
        self.cropped_image = None

        self.set_rect = set_rect  # function
        self.selected_corner = None
        self.scale_factor = 0.5

        self.rect = np.array(rect)

        self.root = tk.Tk()
        self.root.title("Rectangle Adjustment")

        # Lock GUI size to the size of the screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}")

        # Calculate maximum image size based on 2/5 of the window size
        max_width = int(self.root.winfo_screenwidth() * 3 / 5)
        max_height = int(self.root.winfo_screenheight() * 4 / 5)

        self.canvas_original = tk.Canvas(self.root, width=max_width, height=max_height)
        self.canvas_original.pack(side=tk.LEFT, padx=10, pady=10)


        # Determine the scaling factor
        scale_factor_width = max_width / self.image.shape[1]
        scale_factor_height = max_height / self.image.shape[0]
        self.scale_factor = min(scale_factor_width, scale_factor_height)

        # Bind mouse click event to canvas
        self.canvas_original.bind("<Button-1>", self.on_canvas_click)

        # Bind arrow key events to canvas
        self.root.bind("<Left>", self.on_left_arrow)
        self.root.bind("<Right>", self.on_right_arrow)
        self.root.bind("<Up>", self.on_up_arrow)
        self.root.bind("<Down>", self.on_down_arrow)

        # Create a button for saving the image
        self.save_button = tk.Button(self.root, text="Save Rect", command=self.save_rect)
        self.save_button.pack(side=tk.TOP, pady=10)

        # Create a button for loading a new image
        self.load_button = tk.Button(self.root, text="Load New Image", command=self.load_new_image)
        self.load_button.pack(side=tk.TOP, pady=10)

        # Create a window for displaying the cropped image
        self.canvas_transformed = tk.Toplevel(self.root)
        self.initialize_cropped_image_window()

        # Call the draw_rect and update_webcam functions periodically
        self.draw_rect()

    def initialize_cropped_image_window(self):
        # Set the window attributes
        self.canvas_transformed.attributes('-fullscreen', True)  # Set to fullscreen
        # self.canvas_transformed.attributes('-topmost', True)  # Bring to front
        # self.canvas_transformed.attributes('-alpha', 0.7)  # Set transparency (adjust as needed)

        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Set initial position for the second screen (adjust as needed)
        second_screen_x = screen_width  # X-coordinate for the second screen
        second_screen_y = 0  # Y-coordinate for the second screen

        # Set the initial position of the window
        self.canvas_transformed.geometry(f"+{second_screen_x}+{second_screen_y}")

        # Create a canvas for displaying the cropped image
        self.cropped_image_canvas = tk.Canvas(self.canvas_transformed)
        self.cropped_image_canvas.pack(fill=tk.BOTH, expand=tk.YES)

    def draw_rect(self):
        # Draw the original image on the original canvas
        self.transform_and_display()
        # Update the Tkinter window
        self.root.update()

        # Display the cropped_image in the borderless and fullscreen window
        if self.cropped_image is not None:
            cropped_image_rgb = cv2.cvtColor(self.cropped_image, cv2.COLOR_BGR2RGB)
            img_tk_cropped = ImageTk.PhotoImage(image=Image.fromarray(cropped_image_rgb))
            self.cropped_image_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk_cropped)
            self.cropped_image_canvas.image = img_tk_cropped

        # Call the draw_rect function again after a delay (in milliseconds)
        self.root.after(100, self.draw_rect)

    def on_canvas_click(self, event):
        min_distance = float('inf')
        selected_corner = None

        # Calculate distance between mouse click position and each corner
        for i in range(0, len(self.rect), 2):
            x, y = self.rect[i], self.rect[i + 1]
            distance = ((event.x - x) ** 2 + (event.y - y) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                selected_corner = i

        # Select the corner with the smallest distance
        self.selected_corner = selected_corner

    def on_left_arrow(self, event):
        self.adjust_selected_corner(-1, 0)
        self.transform_and_display()

    def on_right_arrow(self, event):
        self.adjust_selected_corner(1, 0)
        self.transform_and_display()

    def on_up_arrow(self, event):
        self.adjust_selected_corner(0, -1)
        self.transform_and_display()

    def on_down_arrow(self, event):
        self.adjust_selected_corner(0, 1)
        self.transform_and_display()

    def adjust_selected_corner(self, delta_x, delta_y):
        if self.selected_corner is not None:
            self.rect[self.selected_corner] += delta_x
            self.rect[self.selected_corner + 1] += delta_y

            # Update the label with the new rectangle parameters
            self.label_var.set(f"Rectangle Parameters: {self.rect}")

    def transform_and_display(self):
        # Transform the image using the specified rectangle
        actual_rect = [int(x / self.scale_factor) for x in self.rect]
        transformed_image = generate_projection(self.image, actual_rect)
        # Display the transformed image on the canvas

        image_rgb = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
        scale_factor = self.scale_factor
        small_height = int(image_rgb.shape[0] * scale_factor)
        small_width = int(image_rgb.shape[1] * scale_factor)
        image_small = cv2.resize(image_rgb, (small_width, small_height))

        img_tk_transformed = ImageTk.PhotoImage(image=Image.fromarray(image_small))
        self.canvas_original.create_image(0, 0, anchor=tk.NW, image=img_tk_transformed)
        self.canvas_original.image = img_tk_transformed
        self.cropped_image = transformed_image

    def save_rect(self):
        actual_rect = [int(x / self.scale_factor) for x in self.rect]
        self.set_rect(actual_rect)
        self.root.destroy()

    def load_new_image(self):
        file_path = filedialog.askopenfilename(title="Select Image File",
                                               filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.image = cv2.imread(file_path)
            if self.image is None:
                raise ValueError(f"Error loading image from path: {file_path}")

            self.rect = [int(self.scale_factor * x) for x in
                         [817, 324, 1186, 329, 1364, 836, 709, 831]]  # Reset rectangle coordinates
            self.selected_corner = None

            # Update the label with the new rectangle parameters
            self.label_var.set(f"Rectangle Parameters: {self.rect}")


def get_projection_rect(image, initial_rect=None):
    return get_rect(image, ProjectionRectApp, initial_rect=initial_rect)


if __name__ == '__main__':
    image_path = r"C:\Users\TLP-299\PycharmProjects\computer-vision-pool\uncropped_images\board1_uncropped.jpg"
    initial_rect = np.array([820, 320, 1100, 300, 1400, 850, 709, 831])  # Initial rectangle coordinates
    image_in = cv2.imread(image_path)
    print(get_projection_rect(image_in, initial_rect=initial_rect))
