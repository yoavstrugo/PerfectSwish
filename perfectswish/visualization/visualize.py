import cv2
import numpy as np

from api.utils import Color, Colors
from perfectswish.api.common_objects import Ball, Board, VelocityVector


def visualize_pool_board(board: Board, path: list[np.array] = None,
                         direction_vectors: list[VelocityVector] = None,
                         balls: list[Ball] = None,
                         path_color: Color = Colors.YELLOW,
                         path_thickness: int = 10, draw_path_junctions: bool = False,
                         junction_color: Color = Colors.YELLOW,
                         junction_radius: int = 5,
                         direction_vector_color: Color = Colors.PURPLE,
                         direction_vector_length: int = 50, direction_vector_thickness: int = 5) -> np.array:
    """
    Creates an image that visualize board with the given path, balls and direction vectors.

    :param board: The board to visualize.
    :param path: The path to visualize.
    :param direction_vectors: The direction vectors to visualize.
    :param balls: The balls to visualize.
    :param path_color: The color of the path.
    :param path_thickness: The thickness of the path.
    :param draw_path_junctions: Whether to draw junctions at each point in the path.
    :param junction_color: The color of the junctions.
    :param junction_radius: The radius of the junctions.
    :param direction_vector_color: The color of the direction vectors.
    :param direction_vector_length: The length of the direction vectors.
    :param direction_vector_thickness: The thickness of the direction vectors.

    :return: The image that visualize the board with the given path, balls and direction vectors.
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
                          color: Color = Colors.PURPLE,
                          length: int = 50, thickness: int = 5) -> None:
    """
    Draws the given velocity vectors on the given image.

    :param board_image: The image of the board.
    :param direction_vectors: The velocity vectors to draw.
    :param color: The color of the velocity vectors.
    :param length: The length of the velocity vectors.
    :param thickness: The thickness of the velocity vectors.
    """
    for vector in direction_vectors:
        cv2.arrowedLine(board_image, pt1=(vector.position[0], vector.position[1]),
                        pt2=(vector.position[0] + int(length * vector.direction[0]),
                             vector.position[1] + int(length * vector.direction[1])),
                        color=color, thickness=thickness)


def draw_balls(board_image: np.array, balls: list[Ball]) -> None:
    """
    Draws the given balls on the given image.

    :param board_image: The image of the board.
    :param balls: The balls to draw.
    """

    for ball in balls:
        cv2.circle(board_image, (ball.position[0], ball.position[1]), ball.radius, ball.color, -1)


def draw_path(board_image: np.array, path: list[np.array], path_color: Color = Colors.YELLOW,
              path_thickness: int = 10, draw_path_junctions: bool = False,
              junction_color: Color = Colors.YELLOW, junction_radius: int = 5) -> None:
    """
    Draws the given path on the given image.

    :param board_image: The image of the board.
    :param path: The path to draw.
    :param path_color: The color of the path.
    :param path_thickness: The thickness of the path.
    :param draw_path_junctions: Whether to draw junctions at each point in the path.
    :param junction_color: The color of the junctions.
    :param junction_radius: The radius of the junctions.
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
