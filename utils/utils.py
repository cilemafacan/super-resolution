#!/usr/bin/env python
# coding: utf-8
import torch
import numpy as np


def RGB2Y(img):

    y = 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
    
    return y

def RGB2YCBCR(img):
    
    y = 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
    cb = 128. + (-37.945 * img[..., 0] - 74.494 * img[..., 1] + 112.439 * img[..., 2]) / 256.
    cr = 128. + (112.439 * img[..., 0] - 94.154 * img[..., 1] - 18.285 * img[..., 2]) / 256.
    
    return np.array([y, cb, cr]).transpose([1, 2, 0])


def YCBCR2RGB(img):
   
    r = 298.082 * img[..., 0] / 256. + 408.583 * img[..., 2] / 256. - 222.921
    g = 298.082 * img[..., 0] / 256. - 100.291 * img[..., 1] / 256. - 208.120 * img[..., 2] / 256. + 135.576
    b = 298.082 * img[..., 0] / 256. + 516.412 * img[..., 1] / 256. - 276.836
   

    return np.array([r, g, b]).transpose([1, 2, 0])

# @func     : preprocess
# @brief    : It turns the y and ycbcr channels by converting the given image to the ycbcr color space.
# @returns  : y and ycbcr channels
# @param    :
#     img   : image to be converted
#    device : GPU or CPU

def preprocess(img, device):

    img = np.array(img).astype(np.float32)
    ycbcr = RGB2YCBCR(img)
    y = ycbcr[..., 0]
    y /= 255.
    y = torch.from_numpy(y).to(device)
    y = y.unsqueeze(0).unsqueeze(0)
    
    return y, ycbcr



