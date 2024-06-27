import cv2
import numpy as np

from perfectswish.utils.utils import Color, Colors
from perfectswish.utils.common_objects import Ball, Board, VelocityVector

from skimage.draw import line_aa

def visualize_pool_board(board: Board, path: list[np.array] = None,
                         direction_vectors: list[VelocityVector] = None,
                         balls: list[Ball] = None,
                         path_color: Color = Colors.GREEN,
                         path_thickness: int = 10, draw_path_junctions: bool = False,
                         junction_color: Color = Colors.GREEN,
                         junction_radius: int = 5,
                         direction_vector_color: Color = Colors.PURPLE,
                         direction_vector_length: int = 50, direction_vector_thickness: int = 5,
                         board_outline_color: Color = Colors.WHITE) -> np.array:
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
    :param board_outline_color: The color of the board outline.

    :return: The image that visualize the board with the given path, balls and direction vectors.
    """

    board_image = draw_board(board=board, board_outline_color=board_outline_color)

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

def draw_board(board: Board, board_outline_color: Color = Colors.WHITE) -> np.array:
    """
    Creates an image that visualize the given board.

    :param board: The board to visualize.

    :return: The image that visualize the given board.
    """
    board_image = np.zeros((board.height, board.width, 3), dtype=np.uint8)
    # add white rectangle outline to represent the board
    cv2.rectangle(board_image, (0, 0), (board.width, board.height), board_outline_color, 10)
    return board_image


def draw_velocity_vectors(board_image: np.array, direction_vectors: list[VelocityVector],
                          color: Color = Colors.BLUE,
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
        cv2.arrowedLine(board_image, pt1=(int(vector.position[0]), int(vector.position[1])),
                        pt2=(int(vector.position[0]) + int(length * vector.direction[0]),
                             int(vector.position[1]) + int(length * vector.direction[1])),
                        color=color, thickness=thickness)


def draw_balls(board_image: np.array, balls: list[Ball]) -> None:
    """
    Draws the given balls on the given image.

    :param board_image: The image of the board.
    :param balls: The balls to draw.
    """

    for ball in balls:
        cv2.circle(board_image, (ball.position[0], ball.position[1]), ball.radius, ball.color, -1)


def draw_path(board_image: np.array, path: list[np.array], path_color: Color = Colors.GREEN,
              path_thickness: int = 10, draw_path_junctions: bool = False,
              junction_color: Color = Colors.GREEN, junction_radius: int = 5) -> None:
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
        # rr, cc, val = line_aa(point[0], point[1], next_point[0], next_point[1])
        # board_image[rr, cc] = val * 255



        # Draw the junctions (circles at each point)
        if draw_path_junctions:
            if i == 0 or i == len(path) - 1:  # Don't draw a junction at the start and end of the path
                continue
            cv2.circle(board_image, (point[0], point[1]), junction_radius, junction_color, -1)



def createLineIterator(P1, P2, img):
    """
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])
    """
    #define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    #difference and absolute difference between points
    #used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    #predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa,dXa),3),dtype=np.float32)
    itbuffer.fill(np.nan)

    #Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X: #vertical line segment
       itbuffer[:,0] = P1X
       if negY:
           itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
       else:
           itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
    elif P1Y == P2Y: #horizontal line segment
       itbuffer[:,1] = P1Y
       if negX:
           itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
       else:
           itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
    else: #diagonal line segment
       steepSlope = dYa > dXa
       if steepSlope:
           slope = dX.astype(np.float32)/dY.astype(np.float32)
           if negY:
               itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
           else:
               itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
           itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(np.uint8) + P1X
       else:
           slope = dY.astype(np.float32)/dX.astype(np.float32)
           if negX:
               itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
           else:
               itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
           itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(np.uint8) + P1Y

    #Remove points outside of image
    colX = itbuffer[:,0]
    colY = itbuffer[:,1]
    itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]

    #Get intensities from img ndarray
    itbuffer[:,2] = img[itbuffer[:,1].astype(np.uint),itbuffer[:,0].astype(np.uint8)]

    return itbuffer