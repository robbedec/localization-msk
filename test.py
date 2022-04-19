import cv2
import pickle

from jinja2 import Undefined
import os
import csv





directory = os.fsencode("data/Database")

image_paths = []

keypoints = []
stop = 0

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    image_paths.append(filename)
    print(filename)

    im=cv2.imread(os.fsdecode(directory) + "/" + filename)
    gr=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    #d=cv2.FeatureDetector_create("SIFT")
    d=cv2.xfeatures2d.SIFT_create(2)
    kp=d.detect(gr)

    index = []
    for point in kp:
        temp = (point.pt, point.size, point.angle, point.response, point.octave, 
            point.class_id) 
        index.append(temp)

    keypoints.append({'id':filename, 'keypoints':index})

    if stop == 5:
        break
    stop+=1


print(keypoints)




with open('src/data/keypoints.csv', mode='w') as csv_file:
    fieldnames = ['id','keypoints']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for key in keypoints:
        writer.writerow(key)
   # writer.writerow({'emp_name': 'Erica Meyers', 'dept': 'IT', 'birth_month': 'March'})# # Dump the keypoints

# with open('src/data/keypoints.txt', 'wb') as handle:
#     pickle.dump(index, handle)



# def read():

#     im=cv2.imread("data/Database/zaal_1__IMG_20190323_111717__01.png")

#     # index = pickle.loads(open("src/data/keypoints.txt").read())

#     index = []
#     with open('src/data/keypoints.txt', 'rb') as handle:
#         index = pickle.load(handle)


#     print(len(index))

#     kp = []

#     for point in index:
#         temp = cv2.KeyPoint(x=point[0][0],y=point[0][1],size=point[1], angle=point[2], 
#                                 response=point[3], octave=point[4], class_id=point[5]) 
#         kp.append(temp)

#     # Draw the keypoints
#     test = Undefined
#     imm = cv2.drawKeypoints(im, keypoints=kp,outImage = None, color=(0,255,0));
#     cv2.imshow("Image", imm);
#     cv2.waitKey(0)


# print(image_paths)

# generate()
# read()

# f = open("src/data/keypoints.txt", "w")
# f.write(pickle.dumps(index))
# f.close()