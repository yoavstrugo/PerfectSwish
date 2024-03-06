import cv2
from perfectswish.object_detection.detect_objects import find_objects as detect
import detect_objects_test_images as test_images

class DetectBalls:
    def __init__(self):
        self.blank = cv2.imread(r"detect_objects_test_images/blank.jpg")
        self.double = cv2.imread(r"detect_objects_test_images/double.jpg")
        self.fifteen = cv2.imread(r"detect_objects_test_images/fifteen1.jpg")
        self.four = cv2.imread(r"detect_objects_test_images/four1.jpg")
        self.green_and_color_near_borders = cv2.imread(r"detect_objects_test_images/green_and_color_near_borders.jpg")
        self.seven = cv2.imread(r"detect_objects_test_images/seven1.jpg")
        self.seven_green_free = cv2.imread(r"detect_objects_test_images/seven_green_free.jpg")
        self.six_green_and_one = cv2.imread(r"detect_objects_test_images/six_green_and_one.jpg")
        self.triangle = cv2.imread(r"detect_objects_test_images/triangle.jpg")
        self.triangle_without_triangle = cv2.imread(r"detect_objects_test_images/triangle_without_triangle.jpg")
        self.triple1 = cv2.imread(r"detect_objects_test_images/triple1.jpg")
        self.triple2 = cv2.imread(r"detect_objects_test_images/triple2.jpg")
        self.triple_and_four_free = cv2.imread(r"detect_objects_test_images/triple_and_four_free.jpg")
        self.processed_images = []

    def return_blank(self):
        image = detect(self.blank, self.blank, True)
        self.processed_images.append(image)


    def balls_together(self):
        image_1 = detect(self.double, self.blank, True)
        image_2 = detect(self.fifteen, self.blank, True)
        image_3 = detect(self.four, self.blank, True)
        image_4 = detect(self.seven, self.blank, True)
        image_5 = detect(self.triangle, self.blank, True)
        image_6 = detect(self.triangle_without_triangle, self.blank, True)
        image_7 = detect(self.triple1, self.blank, True)
        image_8 = detect(self.triple2, self.blank, True)
        image_9 = detect(self.triple_and_four_free, self.blank, True)
        self.processed_images.append(image_1)
        self.processed_images.append(image_2)
        self.processed_images.append(image_3)
        self.processed_images.append(image_4)
        self.processed_images.append(image_5)
        self.processed_images.append(image_6)
        self.processed_images.append(image_7)
        self.processed_images.append(image_8)
        self.processed_images.append(image_9)

    def stripped_balls(self):
        image_1 = detect(self.seven, self.blank, True)
        image_2 = detect(self.triple_and_four_free, self.blank, True)
        self.processed_images.append(image_1)
        self.processed_images.append(image_2)

    def stripped_balls_together(self):
        pass

    def cue_ball_near_stripped_balls(self):
        pass

    def green_balls(self):
        pass

    def shadow(self):
        pass

    def balls_near_border(self):
        pass


    def return_processed_images(self):
        return self.processed_images


class DetectCue:
    def cue_only(self):
        pass
    def cue(self):
        pass

    def cue_near_balls(self):
        pass

    def most_of_cue_outside(self):
        pass

    def balls_in_row_and_cue(self):
        pass

    def balls_in_row_near_border_and_cue(self):
        pass


class DetectMan:
    def man_over_table(self):
        pass


    def man_over_table_with_cue(self):
        pass


    def man_over_table_with_cue_near_balls(self):
        pass



if __name__ == '__main__':
    detect_balls = DetectBalls()
    detect_balls.return_blank()
    detect_balls.balls_together()
    detect_balls.stripped_balls()
    images = detect_balls.return_processed_images()
    for image in images:
        if image is not None:
            cv2.imshow('image', image)
            cv2.waitKey(0)









