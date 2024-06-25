from collections import namedtuple

RESOLUTION_FACTOR = 8
BOARD_BASE_WIDTH = 112
BOARD_BASE_HEIGHT = 224

REAL_BALL_RADIUS_PIXELS = 11

HOLE_RADIUS = 50

Size = namedtuple('Size', ['width', 'height'])
BOARD_SIZE = Size(width=BOARD_BASE_WIDTH * RESOLUTION_FACTOR, height=BOARD_BASE_HEIGHT * RESOLUTION_FACTOR)

DEFAULT_APP_GEOMETRY = "800x600"

__DEBUG_FLAG = False




def set_flag():
    global __DEBUG_FLAG
    __DEBUG_FLAG = True


def get_flag():
    global __DEBUG_FLAG
    if __DEBUG_FLAG:
        __DEBUG_FLAG = False
        return True
    return False
