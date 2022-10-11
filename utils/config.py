import torch
import torch.nn as nn

# Global Defination

scale_factor  = 2

LEARNING_RATE = 0.001
BATCH_SIZE    = 1
EPOCH         = 100
DEVICE        = torch.device("cpu")
LOSS          = nn.MSELoss()
train_dir     = "./input"
val_dir       = "./input"
