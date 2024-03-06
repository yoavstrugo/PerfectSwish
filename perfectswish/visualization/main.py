import cv2
import numpy as np

from perfectswish.api.common_objects import Ball, WhiteBall, Board, VelocityVector
from perfectswish.api.utils import Colors
from perfectswish.visualization.visualize import visualize_pool_board
from perfectswish.simulation import simulate

if __name__ == '__main__':
    board = Board(width=122 * 4, height=244 * 4)
    ball_count = 5
    balls = [WhiteBall(position=np.array([200, 100]), direction=[0.1, 1]),  # White ball
             Ball(position=np.array([500, 400]), color=Colors.YELLOW),
             Ball(position=np.array([400, 500]), color=Colors.RED),
             Ball(position=np.array([300, 300]), color=Colors.BLUE),
             Ball(position=np.array([200, 200]), color=Colors.GREEN),
             Ball(position=np.array([100, 500]), color=Colors.PURPLE)]

    path, hit = simulate.generate_path(balls[0], balls[1:], board, max_len=5)

    direction_vectors = [VelocityVector(position=hit.white_ball_pos, direction=hit.white_ball_hit_vec),
                         VelocityVector(position=hit.ball_pos, direction=hit.ball_hit_vec)]

    image = visualize_pool_board(board=board, balls=balls, path=path, direction_vectors=direction_vectors,
                                 draw_path_junctions=True, path_thickness=5, junction_radius=10,
                                 direction_vector_length=50)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
