import cv2
import os
import csv
import numpy as np
import json



directory = os.fsencode("data/Database")


def generate_csv():
    global directory

    result = []
    #stop = 0

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        print(filename)

        img = cv2.imread(os.fsdecode(directory) + "/" + filename)
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
        #stop+=1


    print(result)

    with open('src/data/keypoints.csv', mode='w') as csv_file:
        fieldnames = ['id','keypoints','descriptors']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for key in result:
            writer.writerow(key)



generate_csv()