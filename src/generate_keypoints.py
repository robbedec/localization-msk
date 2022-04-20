import cv2
import os
import numpy as np
import json
import pandas as pd
import sys



def generate_csv(directory_images,csv_path):
    result = []
    # stop = 0
    # count = 0
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

        parts = filename.split("__")
        photo = parts[1][4:]
        painting_number = int(parts[2][:2])
        
        #print(painting_number)

        result.append({'id':filename, 'keypoints': json.dumps(keypoints), 'descriptors':  json.dumps(descriptors),  'room':  parts[0], 'photo': photo, 'painting_number': painting_number})
        # count += 1
        # if stop == 5:
        #     break
        # stop+=1

    df = pd.DataFrame(result)
    print(df.shape[0])
    df.to_csv(csv_path)  

    # print(count)



if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError('Only provide a path to a video')

    directory_images = os.fsencode(sys.argv[1])   # data/Database
    csv_path = sys.argv[2] # 'src/data/keypoints_2.csv'

    generate_csv(directory_images,csv_path)