import typing
import cv2
import numpy as np
Image = typing.Any

class Image:
    def __init__(self, camera_image: Image):
        self.camera_image = camera_image
        self.calculation_image = None
        self.projection_image = None

    # This method take the image, the user click on four points, and with perspective it will transform the image
    def camera_to_calculation(self, list_of_points): # TODO: Error in reading the image
        # perspective transformation
        cv2.perspectiveTransform(self.camera_image, list_of_points, self.calculation_image)

    def calculate(self):
        pass

    def calculation_to_projection(self, list_of_points):
        # perspective transformation
        pass










