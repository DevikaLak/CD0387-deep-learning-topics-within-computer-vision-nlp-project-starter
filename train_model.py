#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import argparse
import os
import logging
import sys

#Import dependencies for Debugging andd Profiling
import smdebug.pytorch as smd
from smdebug import modes
from smdebug.pytorch import get_hook

#To allow processing of truncated files during train and test cycles
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def  test(model, test_loader, criterion, device, hook):
    '''
    Complete this function that can take a model and a 
    testing data loader and will get the test accuray/loss of the model
    Remember to include any debugging/profiling hooks that you might need
    
    TODO: Add debugging/profiling hooks
    '''
    if hook:
        hook.set_mode(smd.modes.EVAL)
        
    model.eval()
        
    running_loss=0
    running_corrects=0
    
    for data, target in test_loader:
#     for step, (data, target) in enumerate(test_loader):
        
#         dataset_length = len(test_loader.dataset)
#         running_samples = (step + 1) * len(data)
#         proportion = 0.1 # we will use 20% of the dataset
#         if running_samples>(proportion*dataset_length):
#             break
            
        data = data.to(device)
        target = target.to(device)
        outputs = model(data)
        loss = criterion(outputs, target)
        running_loss += loss
        pred = outputs.argmax(dim=1, keepdim=True)
        running_corrects += pred.eq(target.view_as(pred)).sum().item()

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = 100 * (running_corrects / len(test_loader.dataset))

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: ({:.0f}%)\n".format(
            total_loss, total_acc
        )
    )
    
    return total_acc, total_loss

def train(model, train_loader, criterion, optimizer, epochs, device, hook):
    '''
    Complete this function that can take a model and
    data loaders for training and will get train the model
    Remember to include any debugging/profiling hooks that you might need
    
    TODO: Add debugging/profiling hooks
    '''
    if hook:
        hook.set_mode(smd.modes.TRAIN)
    
    model.train()
    
    for e in range(epochs):
        running_loss = 0
        correct = 0
        for data, target in train_loader:
#         for step, (data, target) in enumerate(train_loader):
            
#             dataset_length = len(train_loader.dataset)
#             running_samples = (step + 1) * len(data)
#             proportion = 0.2 # we will use 20% of the dataset
#             if running_samples>(proportion*dataset_length):
#                 break
                
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, target)
            running_loss+=loss
            loss.backward()
            optimizer.step()
            pred = pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        print(f"Epoch {e}: Loss {running_loss/len(train_loader.dataset)}, \
            Accuracy {100*(correct/len(train_loader.dataset))}%")
    
    return model
    
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

def create_data_loaders(train_data_channel, train_batch_size, test_data_channel, test_batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    train_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor()
    ])
    
    train_set = torchvision.datasets.ImageFolder(train_data_channel, transform=train_transform)
    print(f"First image in train is {train_set[0][0]}")
    test_set = torchvision.datasets.ImageFolder(test_data_channel, transform=test_transform)
    print(f"First image in test is {test_set[0][0]}")
    
    
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=True)
    
    return train_loader, test_loader

# save model
def save(model, model_dir):
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)

def main(args):
    
    '''
    Creating a train_loader and test_loader
    '''
    train_data_channel = os.environ['SM_CHANNEL_TRAIN']
    test_data_channel = os.environ['SM_CHANNEL_TEST']
    train_loader, test_loader = create_data_loaders(train_data_channel, args.batch_size, test_data_channel,     args.test_batch_size)
    
    '''
    Initialize a model by calling the net function
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on device {device}")
    
    model=net()
    #Ensures model executes on GPU if available
    model=model.to(device)
    
    '''
    Create your loss and optimizer
    '''
    #Using CrossEntropyLoss as Softmax activation has not been applied in last layer of NN
    loss_criterion = nn.CrossEntropyLoss()
    #Using Adam as optimizer. Not setting lr since Adam uses adaptive learning
    optimizer = optim.Adam(model.parameters(), args.lr)
    
    '''
    Setting hyperparameter
    '''
    epochs = args.epochs
    
    '''
    Creating and Registering hook for bebugging
    '''
    hook = smd.Hook.create_from_json_file()
    if hook:
        hook.register_hook(model)
        print(f"Registering to output loss tensors")
        hook.register_loss(loss_criterion)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    model=train(model, train_loader, loss_criterion, optimizer, epochs, device, hook)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, loss_criterion, device, hook)
    
    '''
    Save the trained model
    '''
    logger.info("SAVE MODEL WEIGHTS")    
    save(model, args.model_dir)

if __name__=='__main__':
    
    parser=argparse.ArgumentParser(description="Dog Breeds Classification")
    '''
    Specify any training args that you might need
    '''
    # Training settings
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 2)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.005, metavar="LR", help="learning rate (default: 0.005)"
    )
    # Container environment
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    # parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    # parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    # parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    # parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    
    args=parser.parse_args()
    
    main(args)
