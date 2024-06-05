import cv2
import numpy as np

from perfectswish.new_image_transformation.frame_decorator import FrameDecorator

class Image_Crop(FrameDecorator):

    def __init__(self, frame, reference_points: np.ndarray = None):
        super().__init__(frame)

        if self._points and (len(self._points) != 4):
            raise ValueError("The frame must have exactly 4 points to transform the image.")

        if reference_points is None:
            self.__reference_points = self.__get_image_corners()

            if not self._points:
                self._set_points(self.__reference_points.copy())

        frame._transform_image.compose(self.__transform_image)
        frame._transform_point.compose(self.__transform_point)

    def __get_image_corners(self):
        """
        This function will return the corners of the canvas (which are the image corners).
        """
        arr = ([
            [0, 0],
            [self._img_orig_width, 0],
            [self._img_orig_width, self._img_orig_height],
            [0, self._img_orig_height]
        ])
        return arr

    def __crop_image(self, image, points):
        """
        This function will crop the image to the points.
        """
        # pass lines between point 1 to 2 to 3 to 4 to 1, and crop it to a rectangle (not tranform, just crop)
        pass
