import torch
import numpy as np
import pandas as pd
import torchvision.models as models
import torchvision.transforms as transforms

from torch.autograd import Variable as V
from PIL import Image

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
            #transforms.Resize((256,256)),
            #transforms.CenterCrop(224),
            transforms.RandomResizedCrop(224),
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
    def __init__(self):
        pass

    @staticmethod
    def create_database(self, database_path):
        # TODO: use custom resnet to extract feature vector of the image
        # combine fvector with keypoint (detected with ORB...).
        pass

if __name__ == '__main__':
    #matcher = PaintingMatcher()
    model = CustomResNet()

    database_folder = '/media/robbedec/BACKUP/ugent/master/computervisie/project/data/Database_paintings/Database'
    #matcher.create_database(database_folder)

    img_path = '/media/robbedec/BACKUP/ugent/master/computervisie/project/data/Database_paintings/Database/zaal_12__IMG_20190323_114236__01.png' 
    img_path = '/media/robbedec/BACKUP/ugent/master/computervisie/project/data/Database_paintings/Database/zaal_13__IMG_20190323_114619__01.png' 
    fvector = model.get_feature_vector(img_path=img_path)
    print(fvector)