import cv2
import numpy as np
from perfectswish.utils import webcam

# initialize a blob detector

params = cv2.SimpleBlobDetector_Params()
# params.filterByArea = True
# params.minArea = 100
# params.filterByCircularity = True
# params.maxCircularity = 0.9
# params.filterByConvexity = True
# params.minConvexity = 0.9
# params.filterByInertia = True
# params.minInertiaRatio = 0.9


blobdetector = cv2.SimpleBlobDetector_create(params)


def find_by_color(image):
    """
    Find the cue stick in the image by color.

    :param image: The image to find the cue stick in.
    :return: The cue stick in the image.
    """
    # Convert the image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


    lower = np.array([100, 100, 5])
    upper = np.array([110, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    # apply mask to original image
    res = cv2.bitwise_and(image, image, mask=mask)

    # erode and dilate the image
    kernel = np.ones((5, 5), np.uint8)
    res = cv2.erode(res, kernel, iterations=6)
    res = cv2.dilate(res, kernel, iterations=3)

    # blur the image
    # res = cv2.GaussianBlur(res, (11, 11), 0)
    res_for_show = cv2.resize(res.copy(), (1280, 720))
    # change the dimensions for the show

    cv2.imshow('mask', res)
    # find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # find the contours diameter
    # filter the contours by area and circularity
    for contour in contours:
        if cv2.contourArea(contour) > 100 and cv2.arcLength(contour, True) > 100:
            cv2.drawContours(res, [contour], -1, (0, 255, 0), 3)
    return res


if __name__ == '__main__':
    # get camera input
    cap = webcam.initialize_webcam(1)

    # save the first frame:
    frame = webcam.get_webcam_image(cap)
    cv2.imwrite('frame.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    while True:
        frame = webcam.get_webcam_image(cap)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cue_stick = find_by_color(frame)
        # lower the image dims
        cue_stick = cv2.resize(cue_stick, (1280, 720))

        cv2.imshow('cue stick', cue_stick)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
