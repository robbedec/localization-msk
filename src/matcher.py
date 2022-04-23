from types import NoneType
import cv2
import os
import numpy as np
import json
import pandas as pd
import sys

import os.path
from os import path



class PaintingMatcher():
    def __init__(self, path=None, directory=None, features=100):
        self.directory = directory

        if path is not None:
            self.load_keypoints(path)
            self.orb = cv2.ORB_create(nfeatures=features)
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            raise ValueError('Path is None.')

    def convert_descriptors(descriptors):
        descriptors = np.array(pd.read_json(descriptors),dtype=np.uint8)
        return descriptors


    def convert_keypoints(keypoint_array):
        keypoints_result = []
        keypoint_array  =  np.array(pd.read_json(keypoint_array))
        for  p in keypoint_array:
            temp = cv2.KeyPoint(
                x=p[0][0],
                y=p[0][1],
                size=p[1],
                angle=p[2],
                response=p[3],
                octave=p[4],
                class_id=p[5],
            )
            keypoints_result.append(temp)
        return keypoints_result
    
    def load_keypoints(self, data_path):

        # if not path.exist(data_path):
        #     raise ValueError('Invalid path.')

        self.df = pd.read_csv(data_path, ",")
        self.df['descriptors'] = self.df['descriptors'].apply(lambda x: PaintingMatcher.convert_descriptors(x))
        self.df['keypoints'] = self.df['keypoints'].apply(lambda x: PaintingMatcher.convert_keypoints(x))

    def  match(self,img_t, display=False):
        #kp_t = self.orb.detect(img_t, None)
        kp_t, des_t = self.orb.detectAndCompute(img_t,  None)

        lowest_distance = 10000000000.0
        index = 0


        distances = []
        if type(des_t) == NoneType:
            return []
        for i, desc in enumerate(self.df['descriptors']):
            matches = self.bf.match(desc, des_t)
            matches = sorted(matches, key = lambda x:x.distance)

            sum = 0
            for m in matches[:10]:
                sum += m.distance

            distances.append((i,sum))
            if sum < lowest_distance:
                lowest_distance = sum
                index= i


        #print(index)

        distances = sorted(distances,key=lambda t:t[1])

        img = cv2.imread(self.directory + "/"+ self.df.id[index], flags = cv2.IMREAD_COLOR)
        # matches = self.bf.match(self.df[self.df.id == name].descriptors[0], des_t)
        matches = self.bf.match(self.df.descriptors[index], des_t)

        matches = sorted(matches, key = lambda x:x.distance)
        result = cv2.drawMatches(img, self.df.keypoints[index], img_t, kp_t, matches[:20], None)

        if(display):
            #cv2.imshow("Query", img_t)
            cv2.namedWindow("result", flags=cv2.WINDOW_NORMAL)
            cv2.imshow("result", result)

            k = cv2.waitKey(0)
            cv2.destroyAllWindows()

        return distances

    def get_filename(self,index):
        return self.df.id[index]

    def get_room(self,index):
        return self.df.room[index]

    def get_photo(self,index):
        return self.df.photo[index]

    def get_painting_number(self,index):
        return self.df.painting_number[index]

if __name__ == '__main__':

    print(sys.argv)

    if len(sys.argv) != 4:
        raise ValueError('Only provide a path to a video')

    path_img = sys.argv[1]
    directory = sys.argv[2]
    path = sys.argv[3]

    matcher = PaintingMatcher(path, directory)

    img = cv2.imread(path_img)        
    print("start")
    # cv2.imshow("Query", img)
    # cv2.waitKey(0)


    result = matcher.match(img)

    print(result)