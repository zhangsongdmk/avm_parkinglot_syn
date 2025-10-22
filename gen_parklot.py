# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 13:27:08 2025

@author: guweihua
"""
import cv2
import os
import random
import numpy as np
from image_shape_change import apply_distortion_fast, apply_wavy_squeeze_fast, apply_distortion
from image_color_change import change_slot_color
import matplotlib.pyplot as plt


def gen_single_rectparklot_mask( size = (50,100), lin_width = 10 , slot_type = 'half_closed_vertical'  ):
    #position = (c_x, c_y, angle)
    #size = (w,h)
    
    bg_width = size[0]
    bg_height = size[1]
    slot_image =  np.ones((bg_height, bg_width,  3), dtype=np.uint8) * 255
    #slot_type ='corner'
    
    if slot_type =='closed':
        pt0_out = (0,0)
        pt1_out = (bg_width,0)
        pt2_out = (bg_width, bg_height)
        pt3_out = (0, bg_height)

        pt0_inner = (lin_width, lin_width)
        pt1_inner = (bg_width - lin_width , lin_width)
        pt2_inner = (bg_width - lin_width , bg_height - lin_width)
        pt3_inner = (lin_width, bg_height- lin_width)

        pts_inner = np.array([pt0_inner, pt1_inner, pt2_inner, pt3_inner], np.int32)
        pts_inner = pts_inner.reshape((-1, 1, 2))
        img_with_polygon = cv2.fillPoly(slot_image, [pts_inner],  color=(0,0,0))
    
    
    elif slot_type == 'half_closed_vertical':
        #
        half_len = bg_width // 4
        pt0_out = (0,0)
        pt1_out = (bg_width,0)
        pt2_out = (bg_width, bg_height)
        pt3_out = (0, bg_height)

        pt0_inner = (lin_width, lin_width)
        pt1_inner = (bg_width - lin_width , lin_width)
        pt2_inner = (bg_width - lin_width , bg_height - lin_width)
        pt3_inner = (lin_width, bg_height- lin_width)
        pts_inner = np.array([pt0_inner, pt1_inner, pt2_inner, pt3_inner], np.int32)
        pts_inner = pts_inner.reshape((-1, 1, 2))
        img_with_polygon = cv2.fillPoly(slot_image, [pts_inner],  color=(0,0,0))

        pt0_inner1 = (half_len, 0)
        pt1_inner1 = ( bg_width - half_len , 0)
        pt2_inner1 = (bg_width - half_len   , bg_height - lin_width)
        pt3_inner1 = (half_len, bg_height- lin_width)
        pts_inner1 = np.array([pt0_inner1, pt1_inner1, pt2_inner1, pt3_inner1], np.int32)
        pts_inner1 = pts_inner1.reshape((-1, 1, 2))
        img_with_polygon = cv2.fillPoly(img_with_polygon, [pts_inner1],  color=(0,0,0))

    elif slot_type == 'half_closed_horizon':
        #
        half_len = bg_height // 4

        pt0_out = (0,0)
        pt1_out = (bg_width,0)
        pt2_out = (bg_width, bg_height)
        pt3_out = (0, bg_height)

        pt0_inner = (lin_width, lin_width)
        pt1_inner = (bg_width - lin_width , lin_width)
        pt2_inner = (bg_width - lin_width , bg_height - lin_width)
        pt3_inner = (lin_width, bg_height- lin_width)
        pts_inner = np.array([pt0_inner, pt1_inner, pt2_inner, pt3_inner], np.int32)
        pts_inner = pts_inner.reshape((-1, 1, 2))
        img_with_polygon = cv2.fillPoly(slot_image, [pts_inner],  color=(0,0,0))

        pt0_inner1 = (0, half_len)
        pt1_inner1 = (bg_width - lin_width, half_len)
        pt2_inner1 = (bg_width - lin_width,   bg_height - half_len  )
        pt3_inner1 = (0, bg_height - half_len)
        pts_inner1 = np.array([pt0_inner1, pt1_inner1, pt2_inner1, pt3_inner1], np.int32)
        pts_inner1 = pts_inner1.reshape((-1, 1, 2))
        img_with_polygon = cv2.fillPoly(img_with_polygon, [pts_inner1],  color=(0,0,0))

    elif slot_type == 'open':  
        #U shape slot
        pt0_out = (0,0)
        pt1_out = (bg_width,0)
        pt2_out = (bg_width, bg_height)
        pt3_out = (0, bg_height)

        pt0_inner = (lin_width, 0)
        pt1_inner = (bg_width - lin_width , 0)
        pt2_inner = (bg_width - lin_width , bg_height - lin_width)
        pt3_inner = (lin_width, bg_height- lin_width)

        pts_inner = np.array([pt0_inner, pt1_inner, pt2_inner, pt3_inner], np.int32)
        pts_inner = pts_inner.reshape((-1, 1, 2))
        img_with_polygon = cv2.fillPoly(slot_image, [pts_inner],  color=(0,0,0))
    elif slot_type == 'open_horizon':  
        #U shape slot
        pt0_out = (0,0)
        pt1_out = (bg_width,0)
        pt2_out = (bg_width, bg_height)
        pt3_out = (0, bg_height)

        pt0_inner = (0, lin_width)
        pt1_inner = (bg_width - lin_width , lin_width)
        pt2_inner = (bg_width - lin_width , bg_height - lin_width)
        pt3_inner = (0, bg_height- lin_width)

        pts_inner = np.array([pt0_inner, pt1_inner, pt2_inner, pt3_inner], np.int32)
        pts_inner = pts_inner.reshape((-1, 1, 2))
        img_with_polygon = cv2.fillPoly(slot_image, [pts_inner],  color=(0,0,0))
    elif slot_type == 'corner':
        #only corner
        half_len = bg_width // 4

        pt0_out = (0,0)
        pt1_out = (bg_width,0)
        pt2_out = (bg_width, bg_height)
        pt3_out = (0, bg_height)

        pt0_inner = (lin_width, lin_width)
        pt1_inner = (bg_width - lin_width , lin_width)
        pt2_inner = (bg_width - lin_width , bg_height - lin_width)
        pt3_inner = (lin_width, bg_height- lin_width)
        pts_inner = np.array([pt0_inner, pt1_inner, pt2_inner, pt3_inner], np.int32)
        pts_inner = pts_inner.reshape((-1, 1, 2))
        img_with_polygon = cv2.fillPoly(slot_image, [pts_inner],  color=(0,0,0))

        pt0_inner1 = (0, half_len)
        pt1_inner1 = (bg_width , half_len)
        pt2_inner1 = (bg_width ,   bg_height - half_len  )
        pt3_inner1 = (0,  bg_height - half_len)
        pts_inner1 = np.array([pt0_inner1, pt1_inner1, pt2_inner1, pt3_inner1], np.int32)
        pts_inner1 = pts_inner1.reshape((-1, 1, 2))
        img_with_polygon = cv2.fillPoly(img_with_polygon, [pts_inner1],  color=(0,0,0))

        pt0_inner1 = ( half_len, 0)
        pt1_inner1 = ( bg_width - half_len , 0)
        pt2_inner1 = ( bg_width - half_len  , bg_height )
        pt3_inner1 = ( half_len, bg_height)
        pts_inner1 = np.array([pt0_inner1, pt1_inner1, pt2_inner1, pt3_inner1], np.int32)
        pts_inner1 = pts_inner1.reshape((-1, 1, 2))
        img_with_polygon = cv2.fillPoly(img_with_polygon, [pts_inner1],  color=(0,0,0))

    slot_image = img_with_polygon
    return slot_image

def rotate_image(image, angle, center=None, scale=1.0):
    """
    旋转整张图像或指定中心点的图像部分。
    
    参数:
        image: 输入图像。
        angle: 旋转的角度，正值表示逆时针方向。
        center: 可选，旋转中心点 (x, y)。默认为图像的中心。
        scale: 缩放因子，默认为1.0，即不缩放。
    返回:
        rotated: 旋转后的图像。
    """
    # 获取图像尺寸
    (h, w) = image.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    # 获取旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    
    # 计算新的边界尺寸
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])

    # 计算新边界宽度和高度
    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)

    # 调整旋转矩阵以考虑平移
    rotation_matrix[0, 2] += bound_w / 2 - center[0]
    rotation_matrix[1, 2] += bound_h / 2 - center[1]

    # 应用仿射变换
    rotated = cv2.warpAffine(image, rotation_matrix, (bound_w, bound_h))

    return rotated

def gen_slantparklot_mask(position = (0,0,0 ), slant_angle = 20, size = (50,100), lin_width = 10, slot_type = 'closed' ):
    #position = (c_x, c_y, angle)
    #size = (w,h)
    
    if slot_type =='closed':
        pass
    elif slot_type == 'open':
        pass
    elif slot_type == 'corner':
        pass





def random_crop_with_mask(image, mask_height, mask_width):
    """
    在图像上使用指定大小的mask随机裁剪一个区域。
    
    参数:
        image: 输入图像 (numpy array, H x W x C)
        mask_height: mask的高度
        mask_width: mask的宽度
    
    返回:
        cropped_image: 裁剪后的图像
    """
    h, w = image.shape[:2]
    
    # 检查mask大小是否小于图像
    if mask_height > h or mask_width > w:
        raise ValueError("Mask size is larger than the image!")
    
    # 随机生成左上角坐标
    # x范围: [0, w - mask_width]
    # y范围: [0, h - mask_height]
    x = random.randint(0, w - mask_width)
    y = random.randint(0, h - mask_height)
    
    # 裁剪图像
    cropped_image = image[y:y + mask_height, x:x + mask_width]
    
    return cropped_image, (x, y, x + mask_width, y + mask_height)  # 返回裁剪区域的坐标


def gen_texture(slot_shape):
    parklot_texturetype = {
        'texture':1,  #
        'pure_color': 
          [(0, 255, 255),  #yellow
              (255,255,255)   #white
          ]
    }
    texture_folder = 'asset/texture/'

    slot_colortype,value  = random.choice(list(parklot_texturetype.items()))
    
    if slot_colortype == 'texture':
        target_texture_file = random.choice(os.listdir(texture_folder))
        target_texture_path = os.path.join(texture_folder, target_texture_file)
        slot_texture = cv2.imread(target_texture_path)
        
        h_t, w_t = slot_texture.shape[:2]
        h_s, w_s = slot_shape.shape[:2]

        if h_t < h_s or w_t < w_s:
            new_width = w_s * 2
            new_height = h_s * 2
            slot_texture  = cv2.resize(slot_texture, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            slot_withcolor = random_crop_with_mask(slot_texture, mask_height = h_s, mask_width = w_s) 
    elif slot_colortype == 'pure_color':
        slot_color = random.choice(value)
        slot_withcolor = slot_color * slot_shape 


    slot_withcolor = slot_withcolor.astype(np.uint8)
    slot_withcolor = change_slot_color(slot_withcolor)

    return slot_withcolor




def mask_shape_augment(slot_mask):
    shape_pipelines = {
    'bigger': 0,      #[0,1,2],    
    'distort':  1  ,    #[0,1]
    'noise':   0,      #[0,1],
    }
    
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
            if value == 1:
                slot_mask_noise = apply_distortion_fast(slot_mask)
                slot_mask_noise = apply_distortion(slot_mask_noise)
            else:
                slot_mask_noise = slot_mask
    
    return slot_shape, slot_mask
    


def overlay_and_crop(background_img, overlay_img, x_offset, y_offset):
    """
    将小图叠加到大图上的指定位置，并裁剪超出大图边缘的部分。    

    """
    # 读取背景图片和需要叠加的小图

    
    bg_height, bg_width = background_img.shape[:2]
    ov_height, ov_width = overlay_img.shape[:2]

    new_image = np.zeros(( bg_height *3, bg_width * 3, background_img.shape[2]))




    ba_ofx = bg_width 
    ba_ofy = bg_height 

    new_image[ba_ofy: ba_ofy + bg_height, ba_ofx:ba_ofx + bg_width ] = background_img

    new_image[ ba_ofy+ y_offset: ba_ofy + y_offset + ov_height,  ba_ofx + x_offset : ba_ofx + x_offset + ov_width ] = overlay_img

    out_image = new_image[ba_ofy: ba_ofy + bg_height, ba_ofx:ba_ofx + bg_width ]
    return out_image

def gen_slot_maskpsv(background_image):
    back_img = np.zeros_like(background_image ) 
    rect_img = np.zeros_like(background_image ) 
 
    bg_height, bg_width = background_image.shape[:2]
    slot_heigh = 500 // 2 
    slot_width = 250 // 2 
    slot_line_w = 20 // 2 

    slot_num = bg_height // slot_width
    slot_positions = []
    offset_ = 10
    for i in range(slot_num):
        slot_position_x =  0  #-slot_heigh  // 3
        slot_position_y = i * (slot_width - slot_line_w) + offset_
        slot_positions.append([slot_position_x, slot_position_y])
    
    for position in slot_positions:
        slot_image = gen_single_rectparklot_mask( size = (slot_width,slot_heigh), lin_width = slot_line_w , slot_type = 'half_closed_vertical'  )
        slot_image = rotate_image(slot_image, -90)

        rect_img = overlay_and_crop(rect_img, slot_image,x_offset= position[0], y_offset= position[1])
    
    rect_img = rotate_image(rect_img, 15)

    out_img = overlay_and_crop (back_img, rect_img,x_offset= - bg_width //4, y_offset= 0  )
    
    return out_img

if __name__ == "__main__":
    img = cv2.imread('data/20161102-161.png')
    bg_height, bg_width = img.shape[:2]
    #tes_img = cv2.resize(img, (bg_height //2, bg_width//2))
    #outimg = overlay_and_crop(img, tes_img, x_offset = 0, y_offset = 0)
    outimg =   gen_slot_maskpsv(img)
    outimg = outimg / 255
    outimg = outimg.astype(np.uint8)
    #gen_slot_maskpsv(img)
    cv2.imwrite('data/multi_slotmask.png',outimg)
    plt.imshow(outimg)
    plt.show()


