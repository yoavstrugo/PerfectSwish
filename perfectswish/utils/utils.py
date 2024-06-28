Color = tuple[int, int, int]


class Colors:
    """
    Predefined colors, in BGR.
    """
    WHITE: Color = (255, 255, 255)
    YELLOW: Color = (0, 255, 255)
    BLUE: Color = (255, 0, 0)
    RED: Color = (0, 0, 255)
    PURPLE: Color = (255, 0, 255)
    ORANGE: Color = (0, 165, 255)
    GREEN: Color = (0, 255, 0)
    BROWN: Color = (42, 42, 165)
    BLACK: Color = (0, 0, 0)
    AQUA: Color = (0, 255, 255)

import cv2
import matplotlib.pyplot as plt
def show_im(image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()