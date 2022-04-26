#from cv2 import find4QuadCornerSubpix
import numpy as np
import cv2
import sys
import math
from matcher import PaintingMatcher
from detector import PaintingDetector
from graph import Graph

from util import (
    generate_graph,
    resize_with_aspectratio,
    random_color,
    order_points,
    rectify_contour
)

class Localiser():
    def __init__(self, matcher=PaintingMatcher("src/data/keypoints.csv","../data/Database"), connectivity_matrix=None) -> None:
        self.matcher = matcher
        self.previous = "Unknown Location"
        #if connectivity_matrix == None:
        #    connectivity_matrix = generate_graph() 
        #self.connectivity_matrix = connectivity_matrix
        #for r in self.connectivity_matrix:
        #    print(r)

    def localise(self, image, contours_list=[], display=False):
        if len(contours_list) == 0:
            return [(self.previous, 0)]

        room_scores = {}
        for contour in contours_list:
            affine_image,crop_img = rectify_contour(contour, image, display=display)
            soft_matches = self.matcher.match(crop_img,display=display)
            print(soft_matches[0:3])
            for m in soft_matches[0:3]:
                room = self.matcher.get_room(m[0])

                # Divide constant by distance; lower distance = bigger number
                # This way, matches with similar distance get similar scores
                room_scores[room] = room_scores.get(room, 0) + 1000/m[1]  
        
        if len(room_scores) == 0:
            return [(self.previous, 0)]
        room_scores_ordered = [(key, room_scores[key]) for key in sorted(room_scores, key=room_scores.get, reverse=True)]
        """
        lowest = 0
        prediction="previous_room"
        for k, v in room_scores.items():
            if(v > lowest):
                prediction = k
                lowest = v
        """

        self.previous = room_scores_ordered[0][0]

        # Of gewoon beste score teruggeven?
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

    room_scores_ordered = localiser.localise(img, contour_results)
    print(room_scores_ordered)
    print("prediction: " + room_scores_ordered[0][0])
    