# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 13:39:29 2025

@author: guweihua
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt



def parklot_compose_posion(forground, background,  forground_position, mask):
    
    for_mask = np.ones_like(background) * 255
    for_mask = for_mask.astype(np.uint8)
    
    #mask_2 = cv2.dilate(mask, kernel, iterations=6).copy()
    
    blended_image = cv2.seamlessClone(forground, background, for_mask, forground_position, cv2.MIXED_CLONE) 
    old_mask = 1
    if old_mask:
        kernel = np.ones((4, 4), np.uint8)
        #alpha = cv2.dilate(mask, kernel, iterations=4) / 255
        alpha = mask / 255
        alpha = cv2.GaussianBlur(alpha , (3, 3), 0)

        #
        if alpha.ndim == 2:
            alpha = np.expand_dims(alpha, axis=2)
        
        
        blended_image = alpha * blended_image * 0.8 + (1 - alpha) * background
    
    blended_image1 = blended_image.astype(np.uint8)
    
    return blended_image1

def parklot_compose_alpha(forground, background,  mask):    
    
        
    alpha = mask / 255 
    alpha = cv2.GaussianBlur(alpha , (3, 3), 0) 
    if alpha.ndim == 2:
        alpha = np.expand_dims(alpha, axis=2)  
    blended_image = alpha * forground * 0.6 + (1 - alpha) * background
    
    blended_image1 = blended_image.astype(np.uint8)
    
    return blended_image1

if __name__ == "__main__":
    forground = cv2.imread('pano004220inmask.png')
    background = cv2.imread('pano004220oimage.jpg')
    forground_position = (100,100)
    mask = forground[:,:,0].copy()
    plt.imshow(mask)
    plt.show()


    result_img,blended_image, mask_2 ,alpha = parklot_compose_posion(forground, background, forground_position, mask)
    plt.imshow(blended_image)
    plt.show()
