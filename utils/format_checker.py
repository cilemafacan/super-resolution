import os
import fnmatch

# @func     : formatCheck
# @brief    : Creates a data list by filtering only images from the given directory.
# @param    :
# directory : directory to filter

suff_check   = ["*.bmp", "*.jpg", "*.jpeg", "*.png", "*.tif", ".tiff"]

def formatCheck(directory):
    
    dataset_list = []

    for suffix in suff_check:

        filtered_data = fnmatch.filter(os.listdir(directory), suffix)
        dataset_list.extend(filtered_data)

    return dataset_list
    
