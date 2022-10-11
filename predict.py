import os
import cv2
import torch
import numpy as np 

from model import SUPRESCNN
from utils.utils import YCBCR2RGB, preprocess
from utils import config, format_checker

# @func     : prediction
# @brief    : It passes the images in the given path through the model and saves the outputs to the output directory.
# @param    :
#     model : trained model
# source_directory   : directory of images to be passed through the model


def prediction(model, source_directory):

    filtered_img_list = format_checker.formatCheck(source_directory)

    for filename in filtered_img_list:

        image_path   = os.path.join(source_directory, filename)
        image        = cv2.imread(image_path)

        image_height = (image.shape[0] // config.scale_factor) * config.scale_factor
        image_width  = (image.shape[1] // config.scale_factor) * config.scale_factor

        image        = cv2.resize(image, (image_width, image_height), cv2.INTER_CUBIC)
        bicubic      = cv2.resize(image, (image_width * config.scale_factor, image_height * config.scale_factor), cv2.INTER_CUBIC)

        image_brigthness_channel , _ = preprocess(image, config.DEVICE)
        _, image_color_channel      = preprocess(bicubic, config.DEVICE)

        with torch.no_grad():
            preds    = model(image_brigthness_channel)

        preds        = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

        out          = np.array([preds, image_color_channel[...,1], image_color_channel[...,2]]).transpose([1,2,0])
        out          = np.clip(YCBCR2RGB(out), 0.0, 255.0).astype(np.uint8)

        if not os.path.exists("output/output_image"):
            
            os.makedirs("output/output_image")
        cv2.imwrite(f"output/output_image/{filename}", out)
