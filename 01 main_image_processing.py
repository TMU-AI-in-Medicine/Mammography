# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 23:30:08 2022

@author: Tommy - Manh - Henry Kha 
"""
import tensorflow as tf
from imutils import paths
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2
import os
from PIL import Image
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from skimage.transform import resize
import imageio

def img_processing(path, ratio, desired_shape, voi_lut = True, fix_monochrome = True):
    dicom = pydicom.read_file(path)
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
    new_width = int(data.shape[0]/ratio)
    
    # Img normalization
    if new_width > data.shape[1]: 
      pad = np.zeros((data.shape[0], new_width - data.shape[1]), dtype=int)
      if dicom.ImageLaterality == "L":
        res = np.concatenate((data, pad), axis=1)
      else:
        res = np.concatenate((pad, data), axis=1)
    else:
        if dicom.ImageLaterality == "L":
            res = data[:,:new_width]
        else:
            res = data[:,-new_width:]
    
    # Resize img  
    res = resize(res,desired_shape)           
    
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        res = np.amax(res) - res
    res = res - np.min(res) 
    res = (np.maximum(res,0)/res.max())*255 # float pixels
    res = np.uint8(res) # integers pixels
    
    return res

ratio = 2048/1664
desired_shape = (2048,1664)
###### HOLOGIC #####
path1 = r"D:\Project AI Mammography\data_mau\ALL_DCM\Selenia_Dimensions_16001626_44_CC_L.dcm"
holo_img = pydicom.dcmread(path1).pixel_array
holo_img.shape
plt.imshow(holo_img,cmap="gray")

holo_processed = img_processing(path1,ratio=ratio,desired_shape=desired_shape)
holo_processed.shape
plt.imshow(holo_processed,cmap="gray")

# SAVE PROCESSED IMG
imageio.imwrite(r"D:\Project AI Mammography\data_mau\ALL_DCM\Selenia_Dimensions_16001626_44_CC_L.dcm.png", holo_processed)

##### GIOTTO #####
path2 = r"D:\Project AI Mammography\data_mau\ALL_DCM\GIOTTO_CLASS_210001508_45_MLO_R.dcm"
giotto_img = pydicom.dcmread(path2).pixel_array
giotto_img.shape
plt.imshow(giotto_img,cmap="gray")

giotto_processed = img_processing(path2,ratio=ratio,desired_shape=desired_shape)
giotto_processed.shape
plt.imshow(giotto_processed,cmap="gray")

# SAVE PROCESSED IMG
imageio.imwrite(r"D:\Project AI Mammography\data_mau\ALL_DCM\GIOTTO_CLASS_210001508_45_MLO_R.dcm.png", giotto_processed)

##### PRISTINA #####

path3 = r"D:\Project AI Mammography\data_mau\ALL_DCM\Senographe_Pristina_2208005625_46_CC_R.dcm"
pristina_img = pydicom.dcmread(path3).pixel_array
pristina_img.shape
plt.imshow(pristina_img,cmap="gray")

pristina_processed = img_processing(path3,ratio=ratio,desired_shape=desired_shape)
pristina_processed.shape
plt.imshow(pristina_processed,cmap="gray")



# SAVE PROCESSED IMG
imageio.imwrite(r"D:\Project AI Mammography\data_mau\ALL_DCM\Senographe_Pristina_2208005625_46_CC_R.dcm.png", pristina_processed)


""" ĐỌC VÀ LƯU HÌNH ẢNH DICOM DƯỚI DẠNG PNG """
def img_processing1(path, fix_monochrome = True):
    dicom = pydicom.read_file(path)
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    data = dicom.pixel_array                      
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    data = data - np.min(data) 
    data = (np.maximum(data,0)/data.max())*255 # float pixels
    data = np.uint8(data) # integers pixels
    return data




""" Chuyển nhiều ảnh cùng lúc"""
root_path = r"D:\Project AI Mammography\data_mau\ALL_DCM"
imagePaths = os.listdir(root_path)
data = []
save_path = r"D:\Project AI Mammography\data_mau\training\training\\"
# loop over the image paths
for imagePath in imagePaths:
    img_path = os.path.join(root_path,imagePath)
    img = img_processing(path=img_path, ratio=ratio, desired_shape=desired_shape)
    imageio.imwrite(os.path.join(save_path,imagePath+".png"),img)
    #data.append(img)
print(data)
plt.imshow(data[25],cmap="gray")


imagePath
os.path.join(save_path,imagePath+".png")
