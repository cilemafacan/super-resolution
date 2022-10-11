import os
import cv2
import h5py
import numpy as np

from utils import config
from utils.format_checker import formatCheck
from utils.utils import RGB2Y


# @func     : trainDataset
# @brief    : It reads the data in the given directory and saves it to the h5 file.
# @param    :
#   dataset : list of data to train

def trainDataset(dataset):
    
    if not os.path.exists("h5_file"):
        os.makedirs("h5_file")

    h5_file        = h5py.File("h5_file/train_h5", "w")
    
    low_res_patch  = []
    high_res_patch = []
    train_dir      = config.train_dir

    try:
        for img in dataset:
    
            img_path        = os.path.join(train_dir, img)
            high_res_img    =  cv2.imread(img_path)
            
            high_res_height = (high_res_img.shape[0] // config.scale_factor) * config.scale_factor
            high_res_width  = (high_res_img.shape[1] // config.scale_factor) * config.scale_factor
            
            high_res        = cv2.resize(high_res_img, (high_res_height, high_res_width), cv2.INTER_CUBIC)
            low_res         = cv2.resize(high_res, (high_res_height // config.scale_factor, high_res_width // config.scale_factor), cv2.INTER_CUBIC)
            
            high_res        = np.array(high_res).astype(np.float32)
            low_res         = np.array(low_res).astype(np.float32)
            
            high_res        = RGB2Y(high_res)
            low_res         = RGB2Y(low_res)
            
            low_res_patch.append(low_res)
            high_res_patch.append(high_res)
    except:
        print(f"{img_path} couldn't read.")  
          
    low_res_patch  = np.array(low_res_patch)
    high_res_patch = np.array(high_res_patch)
    
    h5_file.create_dataset('lr', data=low_res_patch)
    h5_file.create_dataset('hr', data=high_res_patch)

    h5_file.close()

# @func     : valDataset
# @brief    : It reads the data in the given directory and saves it to the h5 file.
# @param    :
#   dataset : list of data to validation

def valDataset(dataset):
    
    h5_file        = h5py.File("h5_file/eval_h5", "w")
    
    low_res_patch  = []
    high_res_patch = []
    val_dir        = config.val_dir

    try:
        for img in dataset:
            
            img_path        = os.path.join(val_dir, img)
            high_res_img    =  cv2.imread(img_path)
            
            high_res_height = (high_res_img.shape[0] // config.scale_factor) * config.scale_factor
            high_res_width  = (high_res_img.shape[1] // config.scale_factor) * config.scale_factor
           
            high_res        = cv2.resize(high_res_img, (high_res_height, high_res_width), cv2.INTER_CUBIC)
            low_res         = cv2.resize(high_res, (high_res_height // config.scale_factor, high_res_width // config.scale_factor), cv2.INTER_CUBIC)
                    
            high_res        = np.array(high_res).astype(np.float32)
            low_res         = np.array(low_res).astype(np.float32)
            
            high_res        = RGB2Y(high_res)
            low_res         = RGB2Y(low_res)

            low_res_patch.append(low_res)
            high_res_patch.append(high_res)
    except:
        print(f"{img_path} couldn'tread.")   
        
    low_res_patch  = np.array(low_res_patch)
    high_res_patch = np.array(high_res_patch)

    h5_file.create_dataset('lr', data=low_res_patch)
    h5_file.create_dataset('hr', data=high_res_patch)

    h5_file.close()

# @func     : datasetCreate
# @brief    : It reads the data in the given directory and checks the format. It then issues the data list to the trainDataset and valDataset 
#             functions to create the h5 file.
# @param    :
# train_dir : train data directory
# val_dir   : validation data directory


def datasetCreate(train_dir, val_dir):

    train_list = formatCheck(train_dir)
    val_list   = formatCheck(val_dir)

    trainDataset(train_list)
    valDataset(val_list)
