#from cv2 import find4QuadCornerSubpix
import numpy as np
import cv2
import sys
import math
from matcher import PaintingMatcher

from util import (
    resize_with_aspectratio,
    random_color,
    order_points,
    rectify_contour
)

class PaintingDetector():
    def __init__(self, img=None, bbox_color=None):
        self._bbox_color = random_color() if bbox_color is None else bbox_color

        if img is not None:
            self.load_image(img)

    def load_image(self, img):
        if not type(img) == np.ndarray:
            raise ValueError()

        # TODO: Rescale image according to its original dimensions.
        # TODO: Calculate blur metric and ignore frame if the score is low.
        # TODO: Dont use a fixed rescale value.
        imh, imw, _ = img.shape
        self._img = resize_with_aspectratio(img, width=500)
        self._original_shape = img.shape

        self._img_bg = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
    
    @property
    def img(self):
        return self._img

    @img.setter
    def img(self, value):
        self.load_image(value)

    def edgemap(self, display=False):
        # Slightly blur the image to reduce noise in the edge detection.
        img_bg_blurred = cv2.GaussianBlur(src=self._img_bg, ksize=(9,9), sigmaX=1)

        # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.402.5899&rep=rep1&type=pdf
        # https://stackoverflow.com/questions/4292249/automatic-calculation-of-low-and-high-thresholds-for-the-canny-operation-in-open
        otsu_thresh_val, _ = cv2.threshold(img_bg_blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        high_thresh_val = otsu_thresh_val
        low_thresh_val = otsu_thresh_val * 0.5

        edgemap = cv2.Canny(image=img_bg_blurred, threshold1=low_thresh_val, threshold2=high_thresh_val, L2gradient=True)

        # Dilate the edgemap to connect
        dilate_kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(1, 3))
        dilated_edgemap = cv2.dilate(src=edgemap, kernel=dilate_kernel, iterations=1)
        dilated_edgemap = cv2.dilate(src=dilated_edgemap, kernel=dilate_kernel.T, iterations=1)

        if display:
            cv2.imshow('Edgemap', edgemap)
            cv2.imshow('Edgemap Dilated', dilated_edgemap)
            cv2.waitKey(0)
        
        return dilated_edgemap
        #return edgemap
    
    """
    Returns a list of contours that qualify as a painting frame (quadrilateral).
    Each contour is given by its four corners. 

    - display:

    Returns a list of candidate contours and the original image annotated with
    the contours. Eeach contour consists of four points and is returned in a
    clock wise manner (Top-left, Top-right, Bottom-right, Bottom-left).
    """
    def contours(self, display=False):
        canny_output = self.edgemap(display=display)
        contour_results = []

        # Find contours and sort them by size. Ideally we only want paintings that are big enough so
        # the details of the painting are visible and usable to apply feature matching in a later stage.
        # This may fail if the contour is too small (TODO: maybe limit to the first X contours). 
        # cv2.RETR_EXTERNAL is supposed to return contours that don't have parents but in practice this
        # not always seem to work. 
        # See https://snippetnuggets.com/howtos/opencv/tips/remove-children-contours-cv2-findContours-only-parents.html
        contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:25]

        # This may be handy later on
        #blob_contours = np.zeros((canny_output.shape[0], canny_output.shape[1], 1), dtype=np.uint8)
        #cv2.fillPoly(blob_contours, pts=contours, color=(255,255,255))

        if display:
            drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

        for i, contour in enumerate(contours):
            # https://stackoverflow.com/a/44156317

            # Generate the convex hull of this contour
            # The returnPoints flag either returns a list of point that form the convex hull (if True).
            # If the flag is False the function returns a list of indices from the original list that
            # indicate the points of the hull.
            convex_hull = cv2.convexHull(points=contour, returnPoints=True)

            # Use approxPolyDP to simplify the convex hull (this should give a quadrilateral for painting frames)
            approx = cv2.approxPolyDP(curve=convex_hull, epsilon=20, closed=True)

            # Ratio of contour area and the convex hull area. This prevents very large and wrong contours.
            # see https://docs.opencv.org/4.x/da/dc1/tutorial_js_contour_properties.html
            solidity = cv2.contourArea(contour) / cv2.contourArea(convex_hull, False)

            # Save the contour if it can be described using a rectangle. The final list contains a list of
            # candidate painting frames.
            # TODO: Ask lecturers on how to find a good general value for this.
            if len(approx) == 4 and solidity > 0.6:
                ordered = order_points(approx.reshape((4,2)))
                contour_results.append(ordered)

            # TODO: Remove this, only used for initial testing
            if display:
                cv2.drawContours(drawing, contours, i, random_color(), 2, cv2.LINE_8, hierarchy, 0)
        
        # Annotate the frame
        original_copy = self._img.copy()
        [ cv2.drawContours(original_copy, contour_results, i, self._bbox_color, 2, cv2.LINE_8, hierarchy, 0) for i in range(len(contour_results)) ]

        # Draw contours
        if display:
            drawing_filtered = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

            # Draw filtered contours on a seperate image.
            [ cv2.drawContours(drawing_filtered, contour_results, i, random_color(), 2, cv2.LINE_8, hierarchy, 0) for i in range(len(contour_results)) ]

            # Show in a window
            cv2.imshow('Original', self._img)
            cv2.imshow('Contours', drawing) # TODO: Remove, only used for initial testing
            cv2.imshow('Contours filtered', drawing_filtered)
            cv2.imshow('Contours filtered on original image', original_copy)
            #cv2.imshow('Blob contours', blob_contours)
            cv2.waitKey(0)
        
        #return contour_results, original_copy
        return self.scale_contour_to_original_coordinates(contour_results,original_copy.shape,self._original_shape), original_copy

    """
    Apply scaling correction to the results. The result is a 3D tensor where the first index is a counter, the other two correspond to the 4x2 representation of 4 2D points.
    Coordinates are scaled to the size of the original image.  
    """
    def scale_contour_to_original_coordinates(self, contour_results, new_shape, original_shape):
        (new_h, new_w, _) = new_shape
        (old_h,  old_w, _) = original_shape  

        scaleY, scaleX = old_h / new_h, old_w / new_w

        return np.array([np.apply_along_axis(lambda row: np.rint(np.multiply(row, [scaleX, scaleY])).astype(int), 1, c) for c in contour_results])
    
if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('Only provide a path to a video')

    impath = sys.argv[1] # Filepath argument
    img = cv2.imread(impath)
    
    detector = PaintingDetector(img)
    contour_results, original_copy = detector.contours(display=False)

    # TODO: dit best allemaal verplaatsen naar een sample file
    # want het wordt snel onoverzichtelijk.
    # Misschien zelfs naar main.py omdat dit werkelijke functionaliteit is.

    #directory_database = '/media/robbedec/BACKUP/ugent/master/computervisie/project/data/Database_paintings/Database'
    #csv_path='/home/robbedec/repos/ugent/computervisie/computervisie-group8/src/data/robbetest.csv'
    #matcher = PaintingMatcher(csv_path, directory_database)
    matcher  = PaintingMatcher("/Users/lennertsteyaert/Documents/GitHub/computervisie-group8/src/data/keypoints.csv","/Users/lennertsteyaert/Documents/GitHub/computervisie-group8/data/Database")
    #matcher  = PaintingMatcher("data/keypoints.csv","../../data/Database")

    
    for i in  range(len(contour_results)):
        affine_image,crop_img = rectify_contour(contour_results[i],img,display=False)
        soft_matches = matcher.match(crop_img,display=False)

        best_match = soft_matches[0]

        room = matcher.get_room(best_match[0])

        photo = matcher.get_photo(best_match[0])

        painting_number = matcher.get_painting_number(best_match[0])
        
        print(f"Room: {room} photo: {photo} number: {painting_number}")