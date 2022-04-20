import cv2
import os
import numpy as np
import json
import pandas as pd
import sys



def generate_csv(directory_images,csv_path):
    result = []
    # stop = 0

    for file in os.listdir(directory_images):
        filename = os.fsdecode(file)
        print(filename)

        img = cv2.imread(os.fsdecode(directory_images) + "/" + filename)
        detector = cv2.ORB_create(nfeatures=100)

        img_keypoints, img_descriptors = detector.detectAndCompute(img,None)

        keypoints = []
        descriptors = []

        for i in range(len(img_keypoints)):
            point = img_keypoints[i]
            descriptor  = img_descriptors[i]
            temp_keypoint = (point.pt, point.size, point.angle, point.response, point.octave, 
                point.class_id) 

            keypoints.append(temp_keypoint)
            descriptors.append(descriptor)


        keypoints = np.array(keypoints).tolist()
        descriptors = np.array(descriptors).tolist()


        result.append({'id':filename, 'keypoints': json.dumps(keypoints), 'descriptors':  json.dumps(descriptors)})

        # if stop == 5:
        #     break
        # stop+=1

    df = pd.DataFrame(result)
    df.to_csv(csv_path)  



if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError('Only provide a path to a video')

    directory_images = os.fsencode(sys.argv[1])   # data/Database
    csv_path = sys.argv[2] # 'src/data/keypoints_2.csv'

    generate_csv(directory_images,csv_path)