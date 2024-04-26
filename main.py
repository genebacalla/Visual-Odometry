import cv2



class VisualOdometry:

    def __init__ (self, blockSize, ksize, k, threshold):
        self.blockSize = blockSize
        self.ksize = ksize
        self.k = k
        self.threshold = threshold

    def __preprocess (self, image):

        img_gauss = cv2.GaussianBlur(image, (5,5), 1)
        img_gray = cv2.cvtColor(img_gauss, cv2.COLOR_BGR2GRAY)

        return img_gray

    def detectCorners(self, image):
        preprocessedImage = self.__preprocess(image) 
        dst = cv2.cornerHarris(preprocessedImage, self.blockSize, self.ksize, self.k)
        norm_corners = cv2.normalize(dst, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        threshold = 0.01 * dst.max()

        return dst, threshold
    
    def drawCorners(self, image):
        image_copy = image.copy()
        dst, thresh = self.detectCorners(image)
        
        image_copy[dst > thresh] = [0,0,255]
        cv2.imshow('corners', image_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

