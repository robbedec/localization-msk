import numpy as np
import cv2
import random as rng

from util import resize_with_aspectratio

class PaintingDetector():
    def __init__(self, img):
        self.load_image(img)

    def load_image(self, img):
        if not type(img) == np.ndarray:
            raise ValueError()

        self._img = resize_with_aspectratio(img, width=800)
        self._img_bg = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
    
    @property
    def img(self):
        return self._img

    @img.setter
    def img(self, value):
        self.load_img(value)

    def edgemap(self, display=False):
        # TODO: Gaussian blur on grayscale image before canny

        # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.402.5899&rep=rep1&type=pdf
        # https://stackoverflow.com/questions/4292249/automatic-calculation-of-low-and-high-thresholds-for-the-canny-operation-in-open
        otsu_thresh_val, _ = cv2.threshold(self._img_bg, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        high_thresh_val = otsu_thresh_val
        low_thresh_val = otsu_thresh_val * 0.5

        edgemap = cv2.Canny(image=self._img_bg, threshold1=low_thresh_val, threshold2=high_thresh_val, L2gradient=True)

        if display:
            cv2.imshow('Edgemap', edgemap)
            cv2.waitKey(0)
        
        return edgemap
    
    """
    Returns a list of contours that qualify as a painting frame (quadrilateral).
    Each contour is given by its four corners. 

    - display:
    """
    def contours(self, display=False):
        canny_output = self.edgemap()
        contour_results = []

        # Find contours and sort them by size. Ideally we only want paintings that are big enough so
        # the details of the painting are visible and usable to apply feature matching in a later stage.
        # This may fail if the contour is too small (TODO: maybe limit to the first X contours). 
        contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

        if display:
            drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
            drawing_filtered = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

        for i, contour in enumerate(contours):
            # https://stackoverflow.com/a/44156317

            # Generate the convex hull of this contour
            # The returnPoints flag either returns a list of point that form the convex hull (if True).
            # If the flag is False the function returns a list of indices from the original list that
            # indicate the points of the hull.
            convex_hull = cv2.convexHull(points=contour, returnPoints=True)

            # Use approxPolyDP to simplify the convex hull (this should give a quadrilateral for painting frames)
            approx = cv2.approxPolyDP(curve=convex_hull, epsilon=20, closed=True)
            # Do something if four corners are found.
            if len(approx) == 4:
                contour_results.append(approx)

            # TODO: Remove this, only used for initial testing
            if display:
                color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
                cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
        
        # Draw contours
        if display:
            # Draw filtered contours on a seperate image.
            for i in range(len(contour_results)):
                color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
                cv2.drawContours(drawing_filtered, contour_results, i, color, 2, cv2.LINE_8, hierarchy, 0)

            # Show in a window
            cv2.imshow('Original', self._img)
            cv2.imshow('Contours', drawing) # TODO: Remove, only used for initial testing
            cv2.imshow('Contours filtered', drawing_filtered)
            cv2.waitKey(0)
        
        return contour_results

if __name__ == '__main__':
    impath = '/media/robbedec/BACKUP/ugent/master/computervisie/project/data/Computervisie 2020 Project Database/test_pictures_msk/20190217_102133.jpg'
    #impath = '/media/robbedec/BACKUP/ugent/master/computervisie/project/data/Computervisie 2020 Project Database/test_pictures_msk/20190217_101930.jpg'
    img = cv2.imread(filename=impath)
    detector = PaintingDetector(img)
    detector.contours(display=True)