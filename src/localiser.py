import numpy as np
import cv2
import sys
import os

from matcher import PaintingMatcher
from detector import PaintingDetector
from hmm import HMM
from preprocessing import FrameProcessor
from util import (
    generate_graph,
    rectify_contour
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Localiser():

    def __init__(self, matcher, graph=None, hmm_distribution='linear') -> None:
        self.matcher = matcher
        self.previous = "..."
        if graph == None:
            graph = generate_graph()
        self.graph = graph
        self.connectivity_matrix = self.graph.getConnectivityMatrix()
        self.hmm = HMM.build(self.connectivity_matrix, hmm_distribution)
    
    def localise(self, image, contours_list=[], display=False, max_room_matches=0):
        if len(contours_list) == 0:
            return self.previous

        dist_list = []
        for contour in contours_list:
            affine_image,crop_img = rectify_contour(contour, image, display=display)
            
            # Don't try to match contour if it's blurry.
            # Results are most likely wrong anyway.
            if FrameProcessor.sharpness_metric(crop_img):
                continue

            soft_matches = self.matcher.match(crop_img,display=True)
            if len(soft_matches) == 0:
                continue
            contour_room_dist = self.getMatchingDistances(soft_matches, max=max_room_matches)
            dist_list.append(contour_room_dist)

        # Return previous result if there are no matches
        if len(dist_list) == 0:
            return self.previous
        
        # Calculate the chance that the frame is located in a room (for every room)
        room_odds = self.calculateRoomOdds(dist_list)

        room_pred = self.hmm.getOptimalPrediction(room_odds, forward=True)
        if room_pred is None or room_pred[1] is None:
            # Geen idee waarom dit gebeurt.
            return self.previous
        self.previous = self.graph.getVertices()[room_pred[1]]
        return self.previous
    
    def calculateRoomOdds(self, distance_list):
        if len(distance_list) > 1:
            for contour_room_dist in distance_list[1:]:
                distance_list[0] += contour_room_dist

        max_dist = max(distance_list[0])
        dist_sum = 0
        for i, val in enumerate(distance_list[0]):
            if val != 0:
                new_val = (max_dist - val + 1)
                distance_list[0][i] = new_val
                dist_sum += new_val
        distance_list[0]/=dist_sum
        return distance_list[0]
            

    def getMatchingDistances(self, soft_matches, max=3):
        if max == 0:
            max = len(self.connectivity_matrix)
        room_dist_list = np.zeros(len(self.connectivity_matrix), np.float32)
        room_count = 0
        idx = 0
        while (room_count < max) & (idx < len(soft_matches)):
            m = soft_matches[idx]
            room = self.matcher.get_room(m[0])
            room_name = room.split("_")[1]
            if(room_name == "V"):   ## Staat niet op grondplan? -> vragen
                idx +=1
                continue

            matrix_index = self.graph.getVertices().index(room_name)
            if room_dist_list[matrix_index] == 0:
                room_dist_list[matrix_index] = m[1]
                room_count +=1
            idx+=1
        return room_dist_list

    @property
    def prob_array(self):
        return self.hmm.prob_arr

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

    localiser = Localiser(hmm_distribution='gaussian')
    location = localiser.localise(img, contour_results)
    print("Zaal: " + location)
    
    #room_scores_ordered = localiser.localise(img, contour_results)
    #print(room_scores_ordered)
    #print("prediction: " + room_scores_ordered[0][0])
    