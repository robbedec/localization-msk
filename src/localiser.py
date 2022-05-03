#from cv2 import find4QuadCornerSubpix
import numpy as np
import cv2
import sys
import math

from torch import matmul
from matcher import PaintingMatcher
from detector import PaintingDetector
from graph import Graph
from scipy import linalg
from hmm import HMM
from util import (
    generate_graph,
    resize_with_aspectratio,
    random_color,
    order_points,
    rectify_contour
)

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Localiser():

    def __init__(self, matcher=PaintingMatcher("src/data/keypoints.csv","../data/Database"), graph=None) -> None:
        self.matcher = matcher
        self.previous = "..."
        if graph == None:
            graph = generate_graph()
        self.graph = graph
        self.connectivity_matrix = self.graph.getConnectivityMatrix()
        self.room_prob = np.ones(len(self.connectivity_matrix))
        self.hmm = HMM.build(self.connectivity_matrix)
    
    def localise_use_all(self, image, contours_list=[], display=False):
        if len(contours_list) == 0:
            return self.previous

        dist_list = []
        for contour in contours_list:
            affine_image,crop_img = rectify_contour(contour, image, display=display)
            soft_matches = self.matcher.match(crop_img,display=True)
            if len(soft_matches) == 0:
                continue
            contour_room_dist = self.getMatchingDistances(soft_matches, max=len(self.connectivity_matrix))
            dist_list.append(contour_room_dist)

        # Return previous result if there are no matches
        if len(dist_list) == 0:
            return self.previous
        
        # Calculate the chance that the frame is located in a room (for every room)
        room_odds = self.calculateRoomOdds(dist_list)

        room_pred = self.hmm.getOptimalPrediction(room_odds)
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
    
    ## Cleanup coming soon :) ##
    """ 
    def localise_v2(self, image, contours_list=[], display=False):
        if len(contours_list) == 0:
            return [self.previous]

        prob_list = []
        for contour in contours_list:
            affine_image,crop_img = rectify_contour(contour, image, display=display)
            soft_matches = self.matcher.match(crop_img,display=display)
            #print(soft_matches[0:3])
            max_dist = 0
            count = 0
            dist_sum = 0
            contour_room_prob = np.zeros(len(self.connectivity_matrix), np.float32)
            for m in soft_matches:

                room = self.matcher.get_room(m[0])

                room_name = room.split("_")[1]
                if(room_name == "V"):
                    continue
                matrix_index = self.graph.getVertices().index(room_name)
                if contour_room_prob[matrix_index] == 0:
                    print(room_name)
                    contour_room_prob[matrix_index] = m[1]
                    count +=1
                    if m[1] > max_dist:
                        max_dist = m[1]
                    if count == 3:
                        break
            for i, val in enumerate(contour_room_prob):
                if val != 0:
                    new_val = (max_dist - val + 1)
                    contour_room_prob[i] = new_val
                    dist_sum += new_val
            contour_room_prob/=dist_sum
            prob_list.append(contour_room_prob)
        if len(prob_list) == 0:
            return [(self.previous, 0)]
        obs_var_matrix = np.array(prob_list).transpose()
        self.calculateHMM(hidden_states, prob_list)
    """
    """
    def calculateHMM(self, hidden_states, obs_var_matrix):
    
        
        P_YX_list = []
        for i, row in enumerate(idx_combinations):
            comb_P_YX = []
            for comb in row:
                P_YX = 1
                for idx in comb:
                    P_YX *= obs_var_matrix[i][idx]
                comb_P_YX.append(P_YX)
            P_YX_list.append(comb_P_YX)
        print(P_YX_list)
        

        for row in idx_cart_product:
            for i, comb in enumerate(row):
                P_YX = obs_var_matrix[i][comb[0]]
                P_X = stat_distr[comb[0]]
                prev = comb[0]
                for idx in comb[1:]:
                    P_YX *= obs_var_matrix[i][idx]
                    P_X *= hidden_states[prev][idx]
                    prev = idx
        
        """ 

        #P_YX = 

        
        #print(stat_distr[0])
        #print(sum(hidden_states[0]*stat_distr[0]))
    """
        test = np.array([[0.5, 0.3, 0.2],
                [0.4, 0.2, 0.4],
                [0.0, 0.3, 0.7]], np.float32)
    """
        
        
        
    """
    def localise(self, image, contours_list=[], display=False):

        if len(contours_list) == 0:
            return [self.previous]

        room_scores = {}
        for contour in contours_list:

            affine_image,crop_img = rectify_contour(contour, image, display=display)
            soft_matches = self.matcher.match(crop_img,display=display)
            print(soft_matches[0:3])
            for m in soft_matches[0:3]:

                room = self.matcher.get_room(m[0])

                room_name = room.split("_")[1]
                if(room_name == "V"):
                    continue
                matrix_index = self.graph.getVertices().index(room_name)
                r = room_scores.get(room, (0, []))
                # Divide constant by distance; lower distance = bigger number
                # This way, matches with similar distance get similar scores
                r = (r[0] + (1000/m[1] * self.room_prob[matrix_index]), r[1])
                r[1].append(m[1])
                room_scores[room] = r
                
        if len(room_scores) == 0:
            return [(self.previous, 0)]
        #room_scores.sort(key=lambda x:x[])
        room_scores_ordered = [(key, room_scores[key][0], room_scores[key][1]) for key in sorted(room_scores, key=lambda x:room_scores[x][0], reverse=True)]
        print(room_scores_ordered)
        if room_scores_ordered[0][1] == 0:
            return [self.previous]
        self.previous = room_scores_ordered[0]
        self.calculateSpatioTemporalLikelihood()
        # Of gewoon beste score teruggeven?
        return room_scores_ordered
    
    def calculateSpatioTemporalLikelihood(self):
        con_matrix = self.graph.getConnectivityMatrix()
        edges = self.graph.getEdges()
        vertices = self.graph.getVertices()
        
        index = vertices.index(self.previous[0].split("_")[1])
        self.room_prob = np.zeros(len(vertices))
        self.room_prob[index] = 1
        for i, edge in enumerate(con_matrix[index]):
            if edge == 1:
                self.room_prob[i] = 0.8
                """

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
    location = localiser.localise_use_all(img, contour_results)
    print("Zaal: " + location)
    
    #room_scores_ordered = localiser.localise(img, contour_results)
    #print(room_scores_ordered)
    #print("prediction: " + room_scores_ordered[0][0])
    