import cv2
import numpy as np

class VisualOdometry:
    def __init__ (self):
        self.max_corners = 25
        self.quality_level = 0.01
        self.min_dst = 10

    def __preprocess (self, image):

        img_gauss = cv2.GaussianBlur(image, (5,5), 1)
        img_gray = cv2.cvtColor(img_gauss, cv2.COLOR_BGR2GRAY)

        return img_gray

    def __detectCorners(self, frame):
        img_gray = self.__preprocess(frame)
        corners = cv2.goodFeaturesToTrack(img_gray, self.max_corners, self.quality_level, self.min_dst)
        corners = np.int0(corners)

        return corners
    
    def drawCorners(self, frame):
        corners = self.__detectCorners(frame)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        return frame



corners = VisualOdometry()
img = cv2.imread('check.jpg')
img_corners = corners.drawCorners(img)
cv2.imshow('corners', img_corners)
cv2.waitKey(0)
cv2.destroyAllWindows()

