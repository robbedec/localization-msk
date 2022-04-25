import cv2
import numpy as np
import pandas as pd
import sys
import os
import json
import torch
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from torch.autograd import Variable as V

# Is dit nog nodig @Lennert?
NoneType = type(None)

class CustomResNet():
    def __init__(self):
        self.model = models.resnet18()
        self.model.eval()
    
    def get_feature_vector(self, img_path):
        # https://towardsdatascience.com/recommending-similar-images-using-pytorch-da019282770c

        feature_layer = self.model.avgpool
        feature_vector = torch.zeros(1, 512, 1, 1)

        # Define image manipulations and process image using standard ResNet parameters.
        img = Image.open(img_path)
        centre_crop = transforms.Compose([
            transforms.Resize((224,224)),
            #transforms.CenterCrop(224),
            #transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        processed_img = V(centre_crop(img).unsqueeze(0))
        
        # Register hook in the forward pass that copies the feature vector out
        # of the Neural Net.
        def copy_hook(m, i, o):
            feature_vector.copy_(o.data)
        h = feature_layer.register_forward_hook(copy_hook)

        # Apply forward pass
        fp = self.model.forward(processed_img)
        
        h.remove()
        return feature_vector.numpy()[0, :, 0, 0]

class PaintingMatcher():
    def __init__(self, path=None, directory=None, features=100):
        self.directory = directory

        if path is not None:
            self.load_keypoints(path)
            self.orb = cv2.ORB_create(nfeatures=features)
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            raise ValueError('Path is None.')
    
    @staticmethod
    def generate_keypoints(directory_images,csv_path):
        neuralnet = CustomResNet()
        result = []

        for file in os.listdir(directory_images):
            filename = os.fsdecode(file)
            print(filename)

            img_path = os.path.join(os.fsdecode(directory_images), filename)
            img = cv2.imread(img_path)
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
            
            result.append({
                'id':filename,
                'keypoints': json.dumps(keypoints),
                'descriptors': json.dumps(descriptors),
                'room':  parts[0],
                'photo': photo,
                'painting_number': painting_number,
                'fvector': json.dumps(neuralnet.get_feature_vector(img_path).tolist())
            })

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

    def match(self,img_t, display=False):
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

        img_path = os.path.join(self.directory, self.df.id[index])
        print(img_path)
        img = cv2.imread(img_path, flags = cv2.IMREAD_COLOR)
        # matches = self.bf.match(self.df[self.df.id == name].descriptors[0], des_t)
        matches = self.bf.match(self.df.descriptors[index], des_t)

        matches = sorted(matches, key = lambda x:x.distance)
        print(len(matches))
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
    cv2.imshow("Query", img)
    cv2.waitKey(0)


    result = matcher.match(img)
    print(result)

    # DO NOT RUN AGAIN
    # Sample to create keypoint file
    #directory_images = os.fsencode(sys.argv[1])   # data/Database
    #csv_path = sys.argv[2] # 'src/data/keypoints_2.csv'
    #matcher = PaintingMatcher.generate_keypoints(directory_images, csv_path)

    #testpad = '/media/robbedec/BACKUP/ugent/master/computervisie/project/data/Database_paintings/Database'
    #directory_images = os.fsencode(testpad)   # data/Database
    #csv_path = '/home/robbedec/repos/ugent/computervisie/computervisie-group8/src/data/robbetest.csv' # 'src/data/keypoints_2.csv'
    #matcher = PaintingMatcher.generate_keypoints(directory_images, csv_path)