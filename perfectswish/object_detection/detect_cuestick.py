import cv2
from cv2 import aruco
import numpy as np
import matplotlib.pyplot as plt

# %%
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
param_markers = aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(marker_dict, param_markers)

MARKER_SIZE = 400


def save_aruco_markers(n=10, filename="markers"):
    # Generating Unique Markers and placing them in a plt grid
    markers = []
    for i in range(n):
        markers.append(aruco.generateImageMarker(marker_dict, i, MARKER_SIZE))

    l = [i for i in range(1, n - 1) if n / i == n // i]  # overkill math to print the markers in a grid
    m = l[len(l) // 2]

    fig, ax = plt.subplots(m, n // m)
    fig.suptitle("Markers")
    # grayscale the plt images
    for i in range(m):
        for j in range(n // m):
            ax[i, j].axis('off')
            ax[i, j].imshow(markers[i * 5 + j], cmap='gray')

    # save the figure
    plt.savefig(filename + ".png")


class CuestickDetector:
    def __init__(self, fiducial_to_stickend_ratio=4 / 9):
        self.marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
        self.param_markers = aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.marker_dict, self.param_markers)
        self.fiducial_to_stickend_ratio = fiducial_to_stickend_ratio

    def detect_cuestick(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        marker_corners, marker_IDs, reject = self.detector.detectMarkers(gray_frame)
        if marker_corners:
            back_fiducial_center = None
            front_fiducial_center = None
            for ids, corners in zip(marker_IDs, marker_corners):
                corners = corners.reshape(4, 2)
                corners = corners.astype(int)
                if ids[0] == 8:
                    back_fiducial_center = np.mean(corners, axis=0)
                elif ids[0] == 9:
                    front_fiducial_center = np.mean(corners, axis=0)
            if back_fiducial_center is not None and front_fiducial_center is not None:
                stickend = back_fiducial_center * (
                        self.fiducial_to_stickend_ratio + 1) - front_fiducial_center * self.fiducial_to_stickend_ratio
                return stickend, back_fiducial_center, front_fiducial_center
        return None

    def draw_cuestick(self, frame, stickend, back_fiducial_center, front_fiducial_center):
        cv2.line(
            frame,
            tuple(back_fiducial_center.astype(int).ravel()),
            tuple(front_fiducial_center.astype(int).ravel()),
            (0, 255, 0),
            4,
            cv2.LINE_AA,
        )

        cv2.circle(frame, tuple(stickend.astype(int).ravel()), 10, (0, 0, 255), -1)

        vector = back_fiducial_center - front_fiducial_center
        vector = vector / np.linalg.norm(vector)
        cv2.arrowedLine(
            frame,
            tuple(stickend.astype(int).ravel()),
            tuple((stickend + vector * 50).astype(int).ravel()),
            (255, 0, 0),
            4,
            cv2.LINE_AA,
        )
        return frame


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    detector = CuestickDetector()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cuestick = detector.detect_cuestick(frame)
        if cuestick is not None:
            stickend, back_fiducial_center, front_fiducial_center = cuestick
            frame = detector.draw_cuestick(frame, stickend, back_fiducial_center, front_fiducial_center)
        cv2.imshow("Cuestick Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
