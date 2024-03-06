import tkinter as tk

import cv2
from PIL import ImageTk, Image


class CalibrationApp:
    def __init__(self, image, set_rect, rect=None, scale_factor=0.4):
        self.root = tk.Tk()
        self.root.title("Adjustment App")

        self.image = image
        self.set_rect = set_rect
        self.scale_factor = scale_factor

        self.root.geometry(f"{int(1200)}x{int(600)}")
        if rect is not None:
            rect = self.scale_factor * rect
        else:
            rect = [int(0.4 * x) for x in [817, 324, 1186, 329, 1364, 836, 709, 831]]
        self.rect = rect

        # Create a button for saving the image
        self.save_image_button = tk.Button(self.root, text="Save Image", command=self.save_rect)
        self.save_image_button.pack(side=tk.TOP, pady=10)

        self.canvas_transformed = tk.Canvas(self.root, width=int(self.image.shape[1] * self.scale_factor),
                                            height=int(self.image.shape[0] * self.scale_factor))
        self.canvas_transformed.pack(side=tk.TOP, padx=10, pady=10,fill="x")

        self.cropped_image = None

        self.root.bind("<Button-1>", self.on_canvas_click)
        self.root.bind("<Left>", self.on_left_arrow)
        self.root.bind("<Right>", self.on_right_arrow)
        self.root.bind("<Up>", self.on_up_arrow)
        self.root.bind("<Down>", self.on_down_arrow)
        self.root.bind("<Escape>", lambda event: self.save_rect())

    def draw_rect(self):
        raise NotImplementedError

    def on_canvas_click(self, event):
        min_distance = float('inf')
        selected_corner = None

        for i in range(0, len(self.rect), 2):
            distance = (event.x - self.rect[i]) ** 2 + (event.y - self.rect[i + 1]) ** 2
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

    def _transformation_func(self, image, rect):
        raise NotImplementedError
    def transform_and_display(self):
        # Transform the image using the specified rectangle
        actual_rect = [int(x / self.scale_factor) for x in self.rect]
        transformed_image = self._transformation_func(self.image, actual_rect)
        # Display the transformed image on the canvas

        image_rgb = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
        scale_factor = self.scale_factor
        small_height = int(image_rgb.shape[0] * scale_factor)
        small_width = int(image_rgb.shape[1] * scale_factor)
        image_small = cv2.resize(image_rgb, (small_width, small_height))

        img_tk_transformed = ImageTk.PhotoImage(image=Image.fromarray(image_small))
        self.canvas_transformed.create_image(0, 0, anchor=tk.NW, image=img_tk_transformed)
        self.canvas_transformed.image = img_tk_transformed
        self.cropped_image = transformed_image

    def save_rect(self):
        actual_rect = [int(x / self.scale_factor) for x in self.rect]
        self.set_rect(actual_rect)
        self.root.destroy()


if __name__ == '__main__':
    path = r"C:\Users\TLP-299\PycharmProjects\PerfectSwish\perfectswish\image_transformation\images\blank_board.jpg"
    image = cv2.imread(path)
    def get_rect(rect):
        print(rect)
    app = CalibrationApp(image, get_rect)
    app.root.mainloop()
