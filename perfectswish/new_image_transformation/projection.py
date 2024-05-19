import cv2
from image_transformation import Image
import tkinter as tk
from tkinter import Canvas
import pickle

from perfectswish.new_image_transformation.projection_gui import MovablePoints


def get_four_points():
    root = tk.Tk()
    app = MovablePoints(root)
    root.mainloop()
    # print the positions
    with open('point_positions.pkl', 'rb') as f:
        positions = pickle.load(f)
    return positions

def main():
    image = Image() # TODO: replace with the image object
    points = get_four_points()
    image.camera_to_calculation(points)

if __name__ == "__main__":
    example_image = cv2.imread("perfect_swish.jpeg")
    image = Image(cv2.imread("chess.jpeg"))
    points = get_four_points(example_image)
    image.calculation_to_projection(points)





