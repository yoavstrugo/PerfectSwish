import cv2
import numpy as np

from perfectswish.new_image_transformation.movable_points_frame import MovablePointsFrame


class ImageTransformFrame(MovablePointsFrame):
    def __init__(self, master, app, get_image, reference_points=None, radius_tolerance=40, fps=24,
                 fine_movement=1):
        """
        This class will allow the user perspective transformation of the image.
        """
        super().__init__(master, app, get_image, reference_points, radius_tolerance, fps, fine_movement)
        # self._points are the points to transform to, they start as the reference points
        if reference_points is None:
            self.__reference_points = self.__get_image_corners()
            self._points = self.__reference_points.copy()
        else:
            self.__reference_points = reference_points

    def __get_image_corners(self):
        """
        This function will return the corners of the canvas (which are the image corners).
        """
        arr = ([
            [0, 0],
            [self._width, 0],
            [self._width, self._height],
            [0, self._height]
        ])
        return arr

    def __get_transformation_matrix(self):
        return cv2.getPerspectiveTransform(np.array(self.__reference_points, dtype=np.float32),
                                           np.array(self._points, dtype=np.float32))

    def _transform_image(self, image):
        """
        This function will transform the image to the new perspective.
        """
        # Get the transformation matrix
        transformation_matrix = self.__get_transformation_matrix()
        # Transform the image
        return cv2.warpPerspective(image, transformation_matrix, (self._width, self._height))

    def _transform_point(self, x, y) -> (int, int):
        """
        This function will transform the point to the new perspective.
        """
        # Get the transformation matrix
        transformation_matrix = self.__get_transformation_matrix()
        # Transform the point
        x, y = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), transformation_matrix)[0][0]
        return x, y
