import cv2
import numpy as np
from typing import Union, Tuple, List

from perfectswish.image_transformation.image_processing import Image
from perfectswish.settings import BOARD_BASE_HEIGHT, BOARD_BASE_WIDTH, RESOLUTION_FACTOR



def find_cue(image: Image):
    crop_rect = [(500, 500), (500, 600), (600,600), (600, 500)]
    target_points = [(0,0), (50, 0), (50, 50), (0, 50)]
    transformation_matrix = cv2.getPerspectiveTransform(np.array(crop_rect, dtype=np.float32), np.array(target_points, dtype=np.float32))
    back_fiducial_center = (490,490)
    front_fiducial_center = (510, 510)
    stickend = (515, 515)
    # show the crop rect
    for i in range(4):
        cv2.line(image, crop_rect[i], crop_rect[(i + 1) % 4], (0, 255, 0), 2)
    # show the cue
    cv2.line(image, back_fiducial_center, front_fiducial_center, (0, 255, 0), 2)
    cv2.line(image, front_fiducial_center, stickend, (0, 255, 0), 2)
    cv2.imshow("image", image)
    transformed_image = cv2.warpPerspective(image, transformation_matrix, (BOARD_BASE_WIDTH, BOARD_BASE_HEIGHT))
    back_fiducial_center_transformed = cv2.perspectiveTransform(np.array([[[back_fiducial_center[0], back_fiducial_center[1]]]], dtype=np.float32), transformation_matrix)[0][0]
    front_fiducial_center_transformed = cv2.perspectiveTransform(np.array([[[front_fiducial_center[0], front_fiducial_center[1]]]], dtype=np.float32), transformation_matrix)[0][0]
    stickend_transformed = cv2.perspectiveTransform(np.array([[[stickend[0], stickend[1]]]], dtype=np.float32), transformation_matrix)[0][0]
    cv2.line(transformed_image, (int(back_fiducial_center_transformed[0]), int(back_fiducial_center_transformed[1])), (int(front_fiducial_center_transformed[0]), int(front_fiducial_center_transformed[1])), (0, 255, 0), 2)
    cv2.line(transformed_image, (int(front_fiducial_center_transformed[0]), int(front_fiducial_center_transformed[1])), (int(stickend_transformed[0]), int(stickend_transformed[1])), (0, 255, 0), 2)
    cv2.imshow("transformed_image", transformed_image)
    cv2.waitKey(0)


if __name__ == "__main__":
    image = cv2.imread(r"C:\Users\TLP\PycharmProjects\PerfectSwish\perfectswish\object_detection\images_test\blank_board.jpg")
    find_cue(image)