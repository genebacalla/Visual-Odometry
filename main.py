import cv2



class VisualOdometry:
    def __init__ (self):
        self.blockSize = 2
        self.ksize = 3
        self.k = 0.04
        self.threshold = 0.01

    def __preprocess (self, image):

        img_gauss = cv2.GaussianBlur(image, (5,5), 1)
        img_gray = cv2.cvtColor(img_gauss, cv2.COLOR_BGR2GRAY)

        return img_gray

    def detectCorners(self, frame):
        
        preprocessedImage = self.__preprocess(frame) 
        dst = cv2.cornerHarris(preprocessedImage, self.blockSize, self.ksize, self.k)
        norm_corners = cv2.normalize(dst, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        threshold = self.threshold * dst.max()

        return dst, threshold
    
    def drawCorners(self, frame):
        frame_copy = frame.copy()
        dst, thresh = self.detectCorners(frame)
        
        frame_copy[dst > thresh] = [0,0,255]
        return frame_copy


corners = VisualOdometry()
img = cv2.imread('check.jpg')
img_corners = corners.drawCorners(img)

cv2.imshow('corners', img_corners)
cv2.waitKey(0)
cv2.destroyAllWindows()

