import cv2
import numpy as np


class Video:
    """
    Video class to ease video object manipulations

    :param video: path to desired video file
    """
    def __init__(self, video):
        # create VideoCapture object from given path
        self.v = cv2.VideoCapture(video)

    def release(self):
        return self.v.release()

    def read(self):
        return self.v.read()

    def is_opened(self):
        return self.v.isOpened()

    def get_fps(self):
        """
        :return: fps of VideoCapture object
        """
        return self.v.get(cv2.CAP_PROP_FPS)
    
    def get_frame_count(self):
        """
        :return: total frame count
        """
        return self.v.get(cv2.CAP_PROP_FRAME_COUNT)
    
    @property
    def len(self):
        """
        :return: length of video in seconds
        """
        return self.get_frame_count() / self.get_fps()
    
    @property
    def shape(self):
        """
        :return: return dimensions of video, width x height
        """
        return int(self.v.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.v.get(cv2.CAP_PROP_FRAME_HEIGHT))


class Line:
    """
    Line class to ease line manipulations
    :param x1: x1 coordinate
    :param y1: y1 coordinate
    :param x2: x2 coordinate
    :param y2: y2 coordinate
    """
    def __init__(self, x1, y1, x2, y2):

        self.x1 = np.float32(x1)
        self.y1 = np.float32(y1)
        self.x2 = np.float32(x2)
        self.y2 = np.float32(y2)

        # calculate slope (direction) and bias
        # np.finfo(float).eps -> number close to 0, but not 0
        self.slope = (self.y2 - self.y1) / (self.x2 - self.x1 + np.finfo(float).eps)
        self.bias = self.y1 - self.slope * self.x1

    def get_coords(self):
        """
        :return: 4 coordinates as a numpy array
        """
        return np.array([self.x1, self.y1, self.x2, self.y2])

    def draw(self, img, thickness=5):
        """
        draws a line on given image

        :param img: image to draw on
        :param thickness: thickness of line
        :return:
        """
        cv2.line(img, (self.x1, self.y1), (self.x2, self.y2), (255, 0, 0), thickness)
