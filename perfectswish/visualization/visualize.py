import cv2
import numpy as np

from perfectswish.api.common_objects import VelocityVector, Ball, Board, Color


def visualize_pool_board(board: Board, path: list[np.array] = None,
                         direction_vectors: list[VelocityVector] = None,
                         balls: list[Ball] = None,
                         path_color: tuple[int, int, int] = Color.YELLOW,
                         path_thickness: int = 10, draw_path_junctions: bool = False,
                         junction_color: tuple[int, int, int] = Color.YELLOW,
                         junction_radius: int = 5,
                         direction_vector_color: tuple[int, int, int] = Color.PURPLE,
                         direction_vector_length: int = 50, direction_vector_thickness: int = 5) -> np.array:
    """
    Creates an image that visualize board with the given path, balls and direction vectors.

    Args:
        board (Board): The board
        path (list[np.array]): The path to visualize (An array of points)
        direction_vectors (list[VelocityVector]): The direction vectors to visualize
        balls (list[Ball]): The balls on the table.

    Returns:
        np.array: The image of the board with the visualization.
    """

    board_image = np.zeros((board.height, board.width, 3), dtype=np.uint8)

    if path:
        draw_path(board_image=board_image, path=path, path_color=path_color,
                  path_thickness=path_thickness, draw_path_junctions=draw_path_junctions,
                  junction_color=junction_color, junction_radius=junction_radius)

    if balls:
        draw_balls(board_image=board_image, balls=balls)

    if direction_vectors:
        draw_velocity_vectors(board_image=board_image, direction_vectors=direction_vectors,
                              color=direction_vector_color, length=direction_vector_length,
                              thickness=direction_vector_thickness)

    return board_image


def draw_velocity_vectors(board_image: np.array, direction_vectors: list[VelocityVector],
                          color: tuple[int, int, int] = Color.PURPLE,
                          length: int = 50, thickness: int = 5) -> None:
    """
    Draws the given velocity vectors on the given image.

    Args:
        board_image (np.array): The image of the board.
        direction_vectors (list[VelocityVector]): The direction vectors to draw.
        color (tuple[int, int, int]): The color of the vectors.
        length (int): The length of the vectors.
        thickness (int): The thickness of the vectors.
    """
    for vector in direction_vectors:
        cv2.arrowedLine(board_image, pt1=(vector.position[0], vector.position[1]),
                        pt2=(vector.position[0] + int(length * vector.direction[0]),
                             vector.position[1] + int(length * vector.direction[1])),
                        color=color, thickness=thickness)


def draw_balls(board_image: np.array, balls: list[Ball]) -> None:
    """
    Draws the given balls on the given image.

    Args:
        board_image (np.array): The image of the board.
        balls (list[Ball]): The balls to draw.
    """

    for ball in balls:
        cv2.circle(board_image, (ball.position[0], ball.position[1]), ball.radius, ball.color, -1)


def draw_path(board_image: np.array, path: list[np.array], path_color: tuple[int, int, int] = Color.YELLOW,
              path_thickness: int = 10, draw_path_junctions: bool = False,
              junction_color: tuple[int, int, int] = Color.YELLOW, junction_radius: int = 5) -> None:
    """
    Draws the given path on the given image.

    Args:
        board_image (np.array): The image of the board.
        path (list[np.array]): The path to draw.
        path_color (tuple[int, int, int]): The color of the path.
        path_thickness (int): The thickness of the path.
        draw_path_junctions (bool): Whether to draw junctions at each point in the path.
        junction_color (tuple[int, int, int]): The color of the junctions.
        junction_radius (int): The radius of the junctions.
    """

    path = [point.astype(int) for point in path]  # Convert the points to integers

    for i in range(len(path) - 1):
        # Draw the path (a line between each point)
        point = path[i]
        next_point = path[i + 1]
        cv2.line(board_image, pt1=(point[0], point[1]), pt2=(next_point[0], next_point[1]),
                 color=path_color,
                 thickness=path_thickness)

        # Draw the junctions (circles at each point)
        if draw_path_junctions:
            if i == 0 or i == len(path) - 2:  # Don't draw a junction at the start and end of the path
                continue
            cv2.circle(board_image, (point[0], point[1]), junction_radius, junction_color, -1)
