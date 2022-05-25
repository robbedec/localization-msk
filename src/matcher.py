from operator import mod
from pyexpat import features
from turtle import mode
import cv2
import numpy as np
import pandas as pd
import sys
import os


import json
# import torch
# import torchvision.models as models
# import torchvision.transforms as transforms
import scipy
import time

from PIL import Image
#from torch.autograd import Variable as V

from util import resize_with_aspectratio
from util import printProgressBar

import tensorflow as tf
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from scipy.spatial import distance
from keras.preprocessing import image


LENNERT = True

# class CustomResNet():
#     def __init__(self):
#         self.model = models.resnet18()
#         self.model.eval()
    
#     def get_feature_vector(self, img_path):
#         # https://towardsdatascience.com/recommending-similar-images-using-pytorch-da019282770c

#         feature_layer = self.model.avgpool
#         feature_vector = torch.zeros(1, 512, 1, 1)

#         # Define image manipulations and process image using standard ResNet parameters.
#         img = Image.open(img_path) if isinstance(img_path, str) else Image.fromarray(img_path)
#         centre_crop = transforms.Compose([
#             # #transforms.Resize((224,224)),
#             # transforms.CenterCrop(224),
#             # #transforms.RandomResizedCrop(224),
#             # transforms.ToTensor(),
#             # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

#             transforms.Resize(224),
#             #transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])
#         processed_img = V(centre_crop(img).unsqueeze(0))
        
#         # Register hook in the forward pass that copies the feature vector out
#         # of the Neural Net.
#         def copy_hook(m, i, o):
#             feature_vector.copy_(o.data)
#         h = feature_layer.register_forward_hook(copy_hook)

#         # Apply forward pass
#         fp = self.model.forward(processed_img) 
        
#         h.remove()
#         return feature_vector.numpy()[0, :, 0, 0]

class CustomResNet():
    def __init__(self):
        self.pretrained_model = VGG16(weights='imagenet', include_top=True)
        self.model = Model(inputs=self.pretrained_model.input, outputs=self.pretrained_model.get_layer("fc2").output)
    
    def get_feature_vector(self, img_path):
        # Reference

        img, x = self.load_image(img_path);
        feat = self.model.predict(x)[0]

        return feat


    def cosine_match(self,img,df):
        img_array = self.preprocess_convert(img)
        vectors = self.model.predict(img_array)[0]

        distances = []
        similar_idx_cosine = [ distance.cosine(vectors, feat) for feat in df["fvector"]]
        idx_closest = sorted(range(len(similar_idx_cosine)), key=lambda k: similar_idx_cosine[k])

        # for i in idx_closest:
        #     print(i)
        for i in idx_closest:
            distances.append((i,similar_idx_cosine[i]))

        return distances

    def load_image(self, path):
        global LENNERT
        if LENNERT:
            img = tf.keras.preprocessing.image.load_img(path, target_size=self.model.input_shape[1:3])
            x = tf.keras.preprocessing.image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            return img, x
        else:
            img = tf.image.load_img(path, target_size=self.model.input_shape[1:3])
            x = tf.keras.utils.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            return img, x

    def preprocess_convert(self, img):
        if LENNERT:
            res = tf.image.resize(img, self.model.input_shape[1:3])
            x = tf.keras.preprocessing.image.img_to_array(res)
            x = np.expand_dims(x, axis=0)
            #x = preprocess_input(x)
            return x        
        else:
            res = tf.image.resize(img, self.model.input_shape[1:3])
            x = tf.keras.utils.img_to_array(res)
            x = np.expand_dims(x, axis=0)
            #x = preprocess_input(x)
            return x        

class PaintingMatcher():
    def __init__(self, path=None, directory=None, features=300, mode = 0):
        self.directory = directory
        self._mode =  mode

        if path is not None:
            self.load_keypoints(path)
            self.orb = cv2.ORB_create(nfeatures=features)
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            raise ValueError('Path is None.')

        self.neuralnet = CustomResNet()    
    

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value

    @staticmethod
    def generate_keypoints(directory_images, csv_path, features=300, fvector_state = True):
        
        neuralnet = None
        if fvector_state:
            neuralnet = CustomResNet()
        result = []

        directory_list = os.listdir(directory_images)
        detector = cv2.ORB_create(nfeatures=features)

        progress = 0
        printProgressBar(progress, len(directory_list), prefix = 'Progress:', suffix = 'Complete', length = 50)

        for file in directory_list:
            filename = os.fsdecode(file)

            img_path = os.path.join(os.fsdecode(directory_images), filename)
            img = cv2.imread(img_path)
            img = resize_with_aspectratio(img, width=800)
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

            keypoints = np.array(keypoints, dtype=object).tolist()
            descriptors = np.array(descriptors, dtype=object).tolist()

            parts = filename.split("__")
            photo = parts[1][4:]
            painting_number = int(parts[2][:2])
            
            if fvector_state:
                result.append({
                    'id':filename,
                    'keypoints': json.dumps(keypoints),
                    'descriptors': json.dumps(descriptors),
                    'room':  parts[0],
                    'photo': photo,
                    'painting_number': painting_number,
                    'fvector': json.dumps(neuralnet.get_feature_vector(img_path).tolist())
                })
            else:
                result.append({
                    'id':filename,
                    'keypoints': json.dumps(keypoints),
                    'descriptors': json.dumps(descriptors),
                    'room':  parts[0],
                    'photo': photo,
                    'painting_number': painting_number,
                    'fvector': json.dumps([])
                })         

            # Update Progress Bar
            progress += 1
            printProgressBar(progress, len(directory_list), prefix = 'Progress:', suffix = 'Complete', length = 50)

        df = pd.DataFrame(result)
        df.to_csv(csv_path)  

    @staticmethod
    def convert_descriptors(descriptors):
        descriptors = np.array(pd.read_json(descriptors), dtype=np.uint8)
        return descriptors

    @staticmethod
    def convert_fvector(fvectors):
        descriptors = np.array(pd.read_json(fvectors), dtype=np.float32)
        return descriptors

    @staticmethod
    def convert_keypoints(keypoint_array):
        keypoints_result = []
        keypoint_array  =  np.array(pd.read_json(keypoint_array), dtype=object)

        if LENNERT:
            for  p in keypoint_array:
                temp = cv2.KeyPoint(
                    x=p[0][0],
                    y=p[0][1],
                    _size=p[1],
                    _angle=p[2],
                    _response=p[3],
                    _octave=p[4],
                    _class_id=p[5],
                )

                keypoints_result.append(temp)
        else:
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
        
        if self._mode != 0:
            self.df['fvector'] = self.df['fvector'].apply(lambda x: PaintingMatcher.convert_fvector(x))

    def match(self,img_t, display=False):
        distances = []
        if(self._mode == 0):
            distances = self.match_mode_orb(img_t,display)
        elif(self._mode == 1):
            distances = self.match_fvector(img_t,display)
        else:
            distances = self.match_combination(img_t,display)

        return distances

    def match_mode_orb(self, img_t, display):

        img_t = resize_with_aspectratio(img_t, width=800)
        kp_t, des_t = self.orb.detectAndCompute(img_t,  None) # Retrieve keypoints and descriptors


        if not type(des_t) == np.ndarray: # Check if any descriptors were returned
            return []
        

        # Distance list has as content (dataframe index, distance score)
        distances = []

        # Loop through the full DB
        for i, desc in enumerate(self.df['descriptors']):
            matches = self.bf.match(desc, des_t) # Retrieve matches for one image in DB
            matches = sorted(matches, key = lambda x:x.distance) # Sort these matches

            sum = 0
            if(len(matches) >= 20): # When more than 20 matches were established all distances are added up
                # Sum of distances (one image 20 best matches)
                for m in matches[:20]:
                    sum += m.distance

                # Add image score to the distance list
                distances.append((i,sum))

        # Sort all DB distance scores
        distances = sorted(distances,key=lambda t: t[1])


        if(display):
            self.show_orb_match(img_t,des_t,kp_t,distances)

        return distances


    def match_fvector(self, img_t, display):
        # Calculate distances for each image in DB (based on fvector)
        current_fvec = self.neuralnet.cosine_match(img_t, self.df)

        if(display):
            self.show_fvector_match(img_t, current_fvec)

        return current_fvec
    
    def show_orb_match(self, img_t, des_t, kp_t,  distances, amount=1):
        for i in range(amount):
            if(len(distances) > i):
                img_path = os.path.join(self.directory, self.df.id[distances[i][0]])
                img = resize_with_aspectratio(cv2.imread(img_path, flags = cv2.IMREAD_COLOR), width=800)
                matches = self.bf.match(self.df.descriptors[distances[i][0]], des_t)
                matches = sorted(matches, key = lambda x:x.distance)
                result = cv2.drawMatches(img, self.df.keypoints[distances[i][0]], img_t, kp_t, matches[:20], None)

                txt = str(distances[i][1])
                cv2.putText(img=result, text=txt, org=(100, 100), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=8, color=(0, 255, 0), thickness=4)
                cv2.namedWindow("Result " + str(i + 1), flags=cv2.WINDOW_NORMAL)
                cv2.imshow("Result " + str(i + 1), result)
            
        cv2.waitKey(1)

    def show_fvector_match(self, img_t, current_fvec, amount=1):
        cv2.namedWindow("Query", flags=cv2.WINDOW_NORMAL)
        res = resize_with_aspectratio(img_t, width=400)
        cv2.imshow("Query", res)

        for i in range(amount):
            if(len(current_fvec) > i):
                img_path = os.path.join(self.directory, self.df.iloc[current_fvec[i][0]].id)
                img = cv2.imread(img_path, flags = cv2.IMREAD_COLOR)
                img = resize_with_aspectratio(img, width=400)
                txt = str(current_fvec[i][1])
                cv2.putText(img=img, text=txt, org=(50, 50), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 255, 0), thickness=2)
                cv2.namedWindow("Resnet " + str(i + 1), flags=cv2.WINDOW_NORMAL)
                cv2.imshow("Resnet " + str(i + 1), img)
            
        cv2.waitKey(1)
    
    def match_combination(self, img_t, display):
        img_t = resize_with_aspectratio(img_t, width=800)
        kp_t, des_t = self.orb.detectAndCompute(img_t,  None) # Retrieve keypoints and descriptors


        if not type(des_t) == np.ndarray: # Check if any descriptors were returned
            return []
        
        # Calculate distances for each image in DB (based on fvector)
        current_fvec = self.neuralnet.cosine_match(img_t, self.df) 

        # Distance list has as content (dataframe index, distance score)
        distances = []

        # Calculate ORB distance only for the first X matches from the fvector matcher
        for el in current_fvec[0:60]:
            desc = self.df.iloc[el[0]].descriptors  # Fetch descriptors

            matches = self.bf.match(desc, des_t)
            matches = sorted(matches, key = lambda x:x.distance)

            sum = 0
            if(len(matches) >= 20): # When more than 20 matches were established all distances are added up  
                # Sum of distances (one image 20 best matches)
                for m in matches[:20]:
                    sum += m.distance

                # Add image score to the distance list
                distances.append((el[0],sum))

        # Sort all DB distance scores
        distances = sorted(distances,key=lambda t: t[1])
        
        if(display):
            self.show_orb_match(img_t,des_t,kp_t,distances)
            self.show_fvector_match(img_t,current_fvec)

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
    if len(sys.argv) != 4:
        raise ValueError('Only provide a path to a video')

    path_img = sys.argv[1]
    directory = sys.argv[2]
    path = sys.argv[3]

    """
    img = cv2.imread(path_img)
    detector = PaintingDetector(img)
    contour_results, original_copy = detector.contours(display=True)
    matcher  = PaintingMatcher("/Users/lennertsteyaert/Documents/GitHub/computervisie-group8/src/data/keypoints.csv","/Users/lennertsteyaert/Documents/GitHub/computervisie-group8/data/Database")
    for i in  range(len(contour_results)):
        affine_image,crop_img = rectify_contour(contour_results[i],img,display=False)
        soft_matches = matcher.match(crop_img,display=False)

        best_match = soft_matches[0]

        room = matcher.get_room(best_match[0])
        photo = matcher.get_photo(best_match[0])
        painting_number = matcher.get_painting_number(best_match[0])
        
        print(f"Room: {room} photo: {photo} number: {painting_number}")
    """

    # DO NOT RUN AGAIN
    # Sample to create keypoint file
    directory_images = os.fsencode(sys.argv[2])   # data/Database
    csv_path = sys.argv[3] # 'src/data/keypoints_2.csv'
    matcher = PaintingMatcher.generate_keypoints(directory_images,csv_path,100) # 100 features