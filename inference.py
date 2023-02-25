import os
import sys
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def net():
    '''
    Complete this function that initializes your model
    Remember to use a pretrained model
    '''
    #Using pretrained ResNet50 model for fine tuning
    model = models.resnet50(pretrained = True)
    
    #Freezing all convolutional layers
    for param in model.parameters():
        param.requires_grad = False
        
    #Extracting the number of activations in the last convolution layer of pretrained model
    num_features = model.fc.in_features
    
    #Adding a fully connected NN to the end. Output is set to 133 is the number of dog breeds in the dataset
    model.fc = nn.Sequential(nn.Linear(num_features, 133))
    
    return model

# TODO: Add model_fn
def model_fn(model_dir):
    logger.info(f"MODEL DIR: {model_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net()
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    model.to(device).eval()
    return model