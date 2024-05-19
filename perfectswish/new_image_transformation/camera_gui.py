from image_transformation import Image
import cv2
import numpy as np
from main import SOURCE

def get_picture_from_camera():
    # This function connect to the camera and take a picture
    camera = cv2.VideoCapture(SOURCE)
    if not camera.isOpened():
        raise Exception("Could not open video device")
    return_value, image = camera.read()
    if not return_value:
        raise Exception("Could not read the image")
    cv2.imread("image.jpg", image)
    print("Image taken")
    camera.release()
    return Image(image)

def gets_four_points(image: Image):
    # This function show the image for to the user, and wait for click.
    # The user will click four times, and the function will mark the points on the image,
    # and will connect the points with lines. after four points, the function will
    # return the list of the points and close the image.
    cv2.imshow("image", image.camera_image)
    points = []
    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            cv2.circle(image.camera_image, (x, y), 5, (0, 0, 255), -1)
            if len(points) > 1:
                cv2.line(image.camera_image, points[-2], points[-1], (0, 0, 255), 2)
            cv2.imshow("image", image.camera_image)
        if len(points) == 4:
            cv2.destroyAllWindows()
    cv2.setMouseCallback("image", click)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return points


def main():
    image = get_picture_from_camera()
    points = gets_four_points(image)
    points = np.array(points, dtype=np.float32)
    image.camera_to_calculation(points)
    cv2.imshow("calculation", image.calculation_image)



if __name__ == "__main__":
    image = Image(cv2.imread("chess.jpeg"))
    points = gets_four_points(image)
    image.camera_to_calculation(points)
    cv2.imshow("calculation", image.calculation_image)




