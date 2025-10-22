# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 15:55:16 2025

@author: guweihua
"""

import random
import cv2
from image_shape_change import apply_distortion_fast, apply_wavy_squeeze_fast, apply_distortion
from image_composting import parklot_compose_posion, parklot_compose_alpha
from image_color_change import change_slot_color
import numpy as np
import matplotlib.pyplot as plt



slot_colors = [(0, 255, 255),  #yellow
              (255,255,255)   #white
              ]
shape_pipelines = {
    'bigger': 0,      #[0,1,2],    
    'distort':  1  ,    #[0,1]
    'noise':   0,      #[0,1],
    }


#we give full pipe here, but can change to random.

def modify_slot_psv(img, mask):

    slot_color = np.array(slot_colors[0] )   
    

        
    slot_mask = mask.copy()
    
    slot_shape = mask.copy()


    slot_shape[slot_shape == 3]  = 0
    slot_shape[slot_shape == 2]  = 0
    slot_mask[slot_mask == 3]  = 0
    slot_mask[slot_mask == 2]  = 0
    for shape_pipeline,value in shape_pipelines.items():
        if shape_pipeline == 'bigger':
            if value == 1:
                kernel = np.ones((4, 4), np.uint8)
                slot_shape = cv2.dilate(slot_mask, kernel, iterations=4)
                slot_mask = cv2.dilate(slot_mask, kernel, iterations=4)
            elif value == 2:
                kernel = np.ones((4, 4), np.uint8)
                slot_shape = cv2.erode(slot_mask, kernel, iterations=4)
                slot_mask = cv2.erode(slot_mask, kernel, iterations=4)                 
        
        if shape_pipeline == 'distort':
            slot_shape = apply_wavy_squeeze_fast(slot_shape)
            slot_mask = apply_wavy_squeeze_fast(slot_mask)
    

        if shape_pipeline == 'noise':
            
            slot_mask_noise = apply_distortion_fast(slot_mask)
            slot_mask_noise = apply_distortion(slot_mask_noise)

    
    

    slot_withcolor = slot_color * slot_shape 
    slot_withcolor = slot_withcolor.astype(np.uint8)
    slot_withcolor = change_slot_color(slot_withcolor)


    background = img
    forground = slot_withcolor  

    background_center = (int(img.shape[0] // 2) , int(img.shape[1] // 2))
    forground_position = background_center
  

    result_img = parklot_compose_posion(forground, background, background_center, slot_mask_noise[:,:,0] *255 )
    #result_img = parklot_compose_alpha(forground, background, slot_mask_noise[:,:,0] *255 )
    #different way of blend parklot.


    result_img = slot_mask * result_img + (1 - slot_mask) * background
    mask[mask == 1] = 0
    new_mask = mask + slot_mask

    return result_img, new_mask

    


if __name__ == "__main__":

    mask = cv2.imread('data/pano004220.png')
    img = cv2.imread('data/pano004220oimage.jpg')
    result_img, new_mask = modify_slot_psv(img, mask)
    cv2.imwrite('result/result_img1.jpg',result_img)
    cv2.imwrite('result/result_mask1.png', new_mask)

    mask = cv2.imread('data/multi_slotmask.png')
    img = cv2.imread('data/20161102-161oimage.jpg')
    result_img, new_mask = modify_slot_psv(img, mask)
    cv2.imwrite('result/20161102-161_new.jpg',result_img)
    cv2.imwrite('result/20161102-161_mask1.png', new_mask)



