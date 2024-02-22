import cv2
import numpy as np

from perfectswish.api.common_objects import Ball, Board, VelocityVector
from perfectswish.api.utils import Colors
from perfectswish.visualization.visualize import visualize_pool_board

if __name__ == '__main__':
    board = Board(width=122 * 4, height=244 * 4)
    ball_count = 5
    balls = [Ball(position=np.array([100, 100])),  # White ball
             Ball(position=np.array([500, 400]), color=Colors.YELLOW),
             Ball(position=np.array([400, 500]), color=Colors.RED),
             Ball(position=np.array([300, 300]), color=Colors.BLUE),
             Ball(position=np.array([200, 200]), color=Colors.GREEN),
             Ball(position=np.array([100, 500]), color=Colors.PURPLE)]

    path = [np.array([100, 100]), np.array([200, 200]), np.array([400, 300]), np.array([300, 100]),
            np.array([400, 100])]

    direction_vectors = [VelocityVector(position=np.array([100, 100]), direction=np.array([1, 1]))]

    image = visualize_pool_board(board=board, balls=balls, path=path, direction_vectors=direction_vectors,
                                 draw_path_junctions=True, path_thickness=5, junction_radius=10,
                                 direction_vector_length=50)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
