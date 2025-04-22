import os
import sys
import cv2
# from skimage import io

import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
# from skimage.transform import rotate
from glob import glob
# from sklearn.model_selection import train_test_split
class CustomDataset(Dataset):
    def __init__(self, data_path, transform, mode='Training', plane=False):
        print("loading data from the directory:", data_path)
        
        # Initialize empty lists to store image and mask paths
        self.image_list = []
        self.image_list_r = []
        self.image_list_g = []  
        self.image_list_b = []  
        self.mask_list = []
        self.folder_names = []
        
        # Walk through the STARCOP directory to find all subfolders
        for folder_name in os.listdir(data_path):
            folder_path = os.path.join(data_path, folder_name)
            
            # Check if it's a directory
            if os.path.isdir(folder_path):
                # Define paths for image and mask
                img_path = os.path.join(folder_path, "mag1c.tif")
                img_path_r  = os.path.join(folder_path, "TOA_AVIRIS_640nm.tif")
                img_path_g  = os.path.join(folder_path, "TOA_AVIRIS_550nm.tif")
                img_path_b  = os.path.join(folder_path, "TOA_AVIRIS_450nm.tif")
                mask_path = os.path.join(folder_path, "label_rgba.tif")
                
                # Check if both files exist
                if os.path.exists(img_path) and os.path.exists(mask_path):
                    self.image_list.append(img_path)
                    self.image_list_r.append(img_path_r)
                    self.image_list_g.append(img_path_g)
                    self.image_list_b.append(img_path_b)
                    self.mask_list.append(mask_path)
                    self.folder_names.append(folder_name)
        
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        
        print(f"Found {len(self.image_list)} valid image-mask pairs")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        """Get the images and masks"""
        img_path = self.image_list[index]
        img_path_r = self.image_list[index]
        img_path_g = self.image_list[index]
        img_path_b = self.image_list[index]
        mask_path = self.mask_list[index]
        folder_name = self.folder_names[index]
        # size_read = 512
        img_mag1c = tifffile.imread(img_path)
        img_mag1c = np.expand_dims(img_mag1c, axis=2)
        img_r = tifffile.imread(img_path_r) 
        img_r = np.expand_dims(img_r, axis=2)
        img_g = tifffile.imread(img_path_g)
        img_g = np.expand_dims(img_g, axis=2)
        img_b = tifffile.imread(img_path_b)
        img_b = np.expand_dims(img_b, axis=2)
        img = np.concatenate([img_r,img_g,img_b,img_mag1c],axis=2)
        label = tifffile.imread(mask_path)
        if self.transform:
            img = self.transform(img)
            label = self.transform(label)
        
        return (img, label)
