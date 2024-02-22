from dataclasses import dataclass

import numpy as np


class Color:
    """
    Represents the color of a ball in the pool game.
    """
    WHITE = (255, 255, 255)
    YELLOW = (0, 255, 255)
    BLUE = (255, 0, 0)
    RED = (0, 0, 255)
    PURPLE = (255, 0, 255)
    ORANGE = (0, 165, 255)
    GREEN = (0, 255, 0)
    BROWN = (42, 42, 165)
    BLACK = (0, 0, 0)


@dataclass
class Ball:
    """
    Represents a ball in the pool game.

    Attributes:
        position (np.array): A 2D vector representing the position of the ball.
        stripped (bool): A boolean representing if the ball is stripped or not.
        radius (int): The radius of the ball.
        color (Color): The color of the ball.
        in_pocket (bool): A boolean representing if the ball is in a pocket or not.
    """
    position: np.array
    stripped: bool = False
    radius: int = 15
    color: Color = Color.WHITE
    in_pocket: bool = False


@dataclass
class WhiteBall(Ball):
    """
    Represents the white ball in the pool game.

    Attributes:
        direction (np.array): A 2D vector representing the direction of the white ball.
    """
    direction: np.array = None


@dataclass
class VelocityVector:
    """
    Represents a velocity vector in the pool game.

    Attributes:
        position (np.array): A 2D vector representing the position of the velocity vector.
        direction (np.array): A 2D vector representing the direction of the velocity vector.
    """
    position: np.array
    direction: np.array


@dataclass
class Cue:
    """
    Represents the cue in the pool game.

    Attributes:
        position (np.array): A 2D vector representing the position of the cue (The tip of the cue).
        direction (np.array): A 2D vector representing the direction of the cue.
    """

    position: np.array
    direction: np.array


@dataclass
class Board:
    """
    Represents the board in the pool game.

    Attributes:
        width (int): The width of the board.
        height (int): The height of the board.
    """
    width: int
    height: int
