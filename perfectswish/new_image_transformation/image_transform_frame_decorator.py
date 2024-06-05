import cv2
import numpy as np

from perfectswish.new_image_transformation.frame_decorator import FrameDecorator


class ImageTransform(FrameDecorator):
    """
    This decorator will apply perspective transformation to the image. The transformation will be from
    the reference points to the points on the frame.
    """

    def __init__(self, frame, reference_points: np.ndarray = None):
        """
        This class will allow the user perspective transformation of the image. Please note that it must
        have exactly 4 points to transform the image. If the frame's points are None, the image corners
        will be selected.
        :param frame: The frame to decorate.
        :param reference_points: The reference points to transform, leave None for the image corners.
        """
        super().__init__(frame)

        # some check
        if self._points and (len(self._points) != 4):
            raise ValueError("The frame must have exactly 4 points to transform the image.")

        if reference_points is None:
            self.__reference_points = self.__get_image_corners()

            if not self._points:
                self._set_points(self.__reference_points.copy())



        # TODO: solve this better, this is bad practice
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



    def __get_transformation_matrix(self):
        """
        This function will return the transformation matrix from the reference points to the points.
        """
        target_points = [self._get_image_point(x, y) for x, y in self._points]
        reference_points = np.array(self.__reference_points, dtype=np.float32)
        target_points = np.array(target_points, dtype=np.float32)
        return cv2.getPerspectiveTransform(reference_points, target_points)

    def __transform_image(self, image: np.ndarray) -> np.ndarray:
        """
        This function will transform the image to the new perspective.
        """
        # Get the transformation matrix
        transformation_matrix = self.__get_transformation_matrix()
        # Transform the image
        return cv2.warpPerspective(image, transformation_matrix, image.shape[:2][::-1])

    def __transform_point(self, x, y) -> (int, int):
        """
        This function will transform the point to the new perspective.
        """
        # Get the transformation matrix
        transformation_matrix = self.__get_transformation_matrix()
        # Transform the point
        x, y = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), transformation_matrix)[0][0]
        return x, y

    def _set_reference_points(self, points: list):
        """
        This function will set the reference points for the transformation.
        """
        self.__reference_points = points
