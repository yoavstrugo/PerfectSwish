import cv2
from cv2 import aruco
import numpy as np
import matplotlib.pyplot as plt

# %%
OFFSET = 3
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
param_markers = aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(marker_dict, param_markers)

MARKER_SIZE = 400


def save_aruco_markers(n=10, filename="markers"):
    # Generating Unique Markers and placing them in a plt grid
    markers = []
    for i in range(n + OFFSET):
        markers.append(aruco.generateImageMarker(marker_dict, i, MARKER_SIZE))
    markers = markers[OFFSET:]

    l = [i for i in range(1, n - 1) if n / i == n // i]  # overkill math to print the markers in a grid
    m = l[len(l) // 2]

    fig, ax = plt.subplots(m, n // m)
    fig.suptitle("Markers")
    # grayscale the plt images

    for i in range(m):
        for j in range(n // m):
            ax[i, j].axis('off')
            ax[i, j].imshow(markers[i * m + j], cmap='gray')

    # save the figure
    plt.savefig(filename + ".png")


class CuestickDetector:
    def __init__(self, fiducial_to_stickend_ratio=4 / 9, back_fiducial_id=8, front_fiducial_id=9):
        self.marker_dict = marker_dict
        self.param_markers = aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.marker_dict, self.param_markers)
        self.fiducial_to_stickend_ratio = fiducial_to_stickend_ratio
        self.back_fiducial_id = back_fiducial_id
        self.front_fiducial_id = front_fiducial_id
        self.back_fiducial_center_coords = np.array([0, 0])
        self.front_fiducial_center_coords = np.array([0, 0])

    def detect_cuestick(self, frame, return_corners=False):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        marker_corners, marker_IDs, reject = self.detector.detectMarkers(gray_frame)
        if marker_corners:
            back_fiducial_center = None
            front_fiducial_center = None
            for ids, corners in zip(marker_IDs, marker_corners):
                corners = corners.reshape(4, 2)
                corners = corners.astype(int)
                if ids[0] == self.back_fiducial_id:
                    back_fiducial_center = np.mean(corners, axis=0)
                elif ids[0] == self.front_fiducial_id:
                    front_fiducial_center = np.mean(corners, axis=0)
            if back_fiducial_center is not None and front_fiducial_center is not None:
                self.back_fiducial_center_coords = back_fiducial_center
                self.front_fiducial_center_coords = front_fiducial_center
                stickend = self.back_fiducial_center_coords * (
                        self.fiducial_to_stickend_ratio + 1) - self.front_fiducial_center_coords * self.fiducial_to_stickend_ratio
                if return_corners:
                    return stickend, self.back_fiducial_center_coords, self.front_fiducial_center_coords, marker_corners
                return stickend, self.back_fiducial_center_coords, self.front_fiducial_center_coords
        return None

    def cover_aruco_markers(self, frame, radius=70):
        cv2.circle(frame, tuple(np.int32(self.back_fiducial_center_coords)), radius, (150, 200, 100), -1)
        cv2.circle(frame, tuple(np.int32(self.front_fiducial_center_coords)), radius, (150, 200, 100), -1)
        return frame

    def draw_cuestick(self, frame):
        # cv2.line(
        #     frame,
        #     tuple(self.back_fiducial_center_coords.astype(int).ravel()),
        #     tuple(self.front_fiducial_center_coords.astype(int).ravel()),
        #     (0, 255, 0),
        #     4,
        #     cv2.LINE_AA,
        # )

        stickend = self.back_fiducial_center_coords * (
                self.fiducial_to_stickend_ratio + 1) - self.front_fiducial_center_coords * self.fiducial_to_stickend_ratio
        cv2.circle(frame, tuple(stickend.astype(int).ravel()), 10, (0, 0, 255), -1)

        vector = self.back_fiducial_center_coords - self.front_fiducial_center_coords
        if np.linalg.norm(vector) != 0:
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
    detector = CuestickDetector(back_fiducial_id=OFFSET, front_fiducial_id=OFFSET + 1)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cuestick = detector.detect_cuestick(frame)
        frame = detector.draw_cuestick(frame)
        covered_frame = detector.cover_aruco_markers(frame)
        cv2.imshow("Cuestick Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# if __name__ == '__main__':
#     save_aruco_markers(n=4, filename="large_markers")
#
