import os
import time
import numpy as np

import torch
import torch.optim as optim

from torch import nn
from tqdm import tqdm
from model import SUPRESCNN

# @func     : train
# @brief    : Returns the loss values ​​by training the data with the created model
# @return   : Loss
# @param    :
#       net : model for training
#    epoch  : number of epochs to train
#   device  : device for training. GPU or CPU
#   dataset : dataset to train
# dataloader: data loader that loads data a certain number of times and mixed
# optimizer : optimization function to be used
# criterion : loss function to calculate the loss


def train(net, dataset, dataloader, device, criterion, optimizer, epoch):
    
    net.train()
    running_loss = 0.0
   
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
        
        low_res, high_res = data
        low_res             = low_res.to(device)
        high_res            = high_res.to(device)
        optimizer.zero_grad()

        output              = net(low_res)
        loss                = criterion(output, high_res)

        loss.backward()
        optimizer.step()

        running_loss      += loss.item()
        
    train_loss =  running_loss/len(dataloader.dataset)
    print(f"Train Loss: {train_loss}")
    
    return train_loss


# @func     : validation
# @brief    : Tests the accuracy of the training and returns the loss.
# @return   : Loss
# @param    :
#       net : model for validation
#    epoch  : number of epochs to validation
#   device  : device for validation. GPU or CPU
#   dataset : dataset to validation
# dataloader: data loader that loads data a certain number of times and mixed
# optimizer : optimization function to be used
# criterion : loss function to calculate the loss


def validation(net,dataset, dataloader, device, criterion, optimizer, epoch):
    
    net.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
            
            low_res, high_res = data
            low_res             = low_res.to(device)
            high_res            = high_res.to(device)
            output              = net(low_res)
            loss                = criterion(output, high_res)
            running_loss       += loss.item()
            
        val_loss = running_loss/len(dataloader.dataset)
        print(f"Val Loss: {val_loss}")
        
        return val_loss
            


