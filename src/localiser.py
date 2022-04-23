#from cv2 import find4QuadCornerSubpix
import numpy as np
import cv2
import sys
import math
from matcher import PaintingMatcher
from detector import PaintingDetector

from util import (
    resize_with_aspectratio,
    random_color,
    order_points,
    rectify_contour
)

class Localiser():
    def __init__(self, matcher=PaintingMatcher("src/data/keypoints.csv","../data/Database")) -> None:
        self.matcher = matcher

    def localise(self, contours_list=[]):
        if len(contours_list) == 0:
            return "previous room"

        room_scores = {}
        for contour in  contours_list:
            affine_image,crop_img = rectify_contour(contour,img,display=False)
            soft_matches = self.matcher.match(crop_img,display=True)
            
            for m in soft_matches:
                room = self.matcher.get_room(m[0])

                # Divide constant by distance, lower distance = bigger number
                # This way, matches with similar distance get similar scores
                room_scores[room] = room_scores.get(room, 0) + 100/m[1]  
    
        room_scores_ordered = [(key, room_scores[key]) for key in sorted(room_scores, key=room_scores.get, reverse=True)]
        """
        lowest = 0
        prediction="previous_room"
        for k, v in room_scores.items():
            if(v > lowest):
                prediction = k
                lowest = v
        """
        return room_scores_ordered


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('Only provide a path to an image')

    # Filepath argument 
    # e.g: ../data/Computervisie 2020 Project Database/dataset_pictures_msk/zaal_19/IMG_20190323_121333.jpg
    impath = sys.argv[1] 

    img = cv2.imread(filename=impath)

    detector = PaintingDetector(img)
    contour_results, original_copy = detector.contours(display=False)
    #contour_results_rescaled = detector.scale_contour_to_original_coordinates(contour_results,original_copy.shape,img.shape)

    localiser = Localiser()
    room_scores_ordered = localiser.localise(contour_results)
    print(room_scores_ordered)
    print("prediction: " + room_scores_ordered[0][0])