# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 13:47:58 2025

@author: guweihua
"""

import cv2
import numpy as np

def apply_distortion(image, grid_size=10, max_shift=10):
    """
    对图像应用局部随机扭曲效果
    :param image: 输入图像 (numpy array)
    :param grid_size: 网格大小（每块的像素尺寸）
    :param max_shift: 最大随机偏移量（控制扭曲强度）
    :return: 扭曲后的图像
    """
    h, w = image.shape[:2]
    distorted = np.zeros_like(image)

    # 创建输出图像的坐标网格
    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            # 定义当前块的区域
            y_end = min(y + grid_size, h)
            x_end = min(x + grid_size, w)

            # 随机偏移（模拟扭曲）
            dx = np.random.randint(-max_shift, max_shift)
            dy = np.random.randint(-max_shift, max_shift)

            # 新位置（带边界检查）
            new_y = y + dy
            new_x = x + dx
            new_y_end = new_y + (y_end - y)
            new_x_end = new_x + (x_end - x)

            # 裁剪新位置到图像范围内
            src = image[y:y_end, x:x_end]

            # 计算在目标图像中的有效区域
            dest_y_start = max(new_y, 0)
            dest_y_end = min(new_y_end, h)
            dest_x_start = max(new_x, 0)
            dest_x_end = min(new_x_end, w)

            # 对应源图像中要复制的区域（考虑裁剪）
            src_y_start = dest_y_start - new_y + (y - y)
            src_y_end = src_y_start + (dest_y_end - dest_y_start)
            src_x_start = dest_x_start - new_x + (x - x)
            src_x_end = src_x_start + (dest_x_end - dest_x_start)

            if src_y_end > src_y_start and src_x_end > src_x_start:
                distorted[dest_y_start:dest_y_end, dest_x_start:dest_x_end] = \
                    src[src_y_start:src_y_end, src_x_start:src_x_end]

    return distorted

def apply_distortion_fast(image):
    h_i, w_i = image.shape[:2]  
    
    
    h = h_i 
    w = w_i 
    distorted = np.zeros_like(image) 
    dx_map = np.random.randn(h, w) * 5
    dy_map = np.random.randn(h, w) *5
    map_x = (np.arange(w) + dx_map).astype(np.float32)
    
    map_y = (np.arange(h)[:, None] + dy_map).astype(np.float32)
    
    
    #map_x = cv2.resize(map_x, (h_i,w_i))
    #map_y = cv2.resize(map_y, (h_i,w_i))
   
    distorted = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    return distorted



def apply_wavy_squeeze(image, wave_amp=5, wave_freq=0.05, squeeze_strength=0.3):
    """
    应用波浪形挤压效果
    :param image: 输入图像 (H x W x C)
    :param wave_amp: 波浪振幅（控制波浪高度）
    :param wave_freq: 波浪频率（控制波浪密度，值越大波越多）
    :param squeeze_strength: 挤压强度（控制整体挤压程度，0~1）
    :return: 扭曲后的图像
    """
    h, w = image.shape[:2]

    # 创建坐标网格
    x_map = np.zeros((h, w), dtype=np.float32)
    y_map = np.zeros((h, w), dtype=np.float32)

    # 生成波浪形坐标映射
    for y in range(h):
        for x in range(w):
            # 水平方向波浪挤压：X 坐标受 Y 影响（垂直波纹）
            offset_x = wave_amp * np.sin(2 * np.pi * wave_freq * y) * (1 - squeeze_strength * (x / w))
            x_map[y, x] = x + offset_x

            # 垂直方向波浪挤压：Y 坐标受 X 影响（水平波纹）
            offset_y = wave_amp * np.sin(2 * np.pi * wave_freq * x) * (1 - squeeze_strength * (y / h))
            y_map[y, x] = y + offset_y

    # 使用 remap 重映射图像
    distorted = cv2.remap(image, x_map, y_map, interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)  # BORDER_REFLECT 减少黑边

    return distorted

def apply_wavy_squeeze_fast(image, wave_amp=2, wave_freq=0.01, squeeze_strength=0.3):
    """
    应用波浪形挤压效果
    :param image: 输入图像 (H x W x C)
    :param wave_amp: 波浪振幅（控制波浪高度）
    :param wave_freq: 波浪频率（控制波浪密度，值越大波越多）
    :param squeeze_strength: 挤压强度（控制整体挤压程度，0~1）
    :return: 扭曲后的图像
    """
    h, w = image.shape[:2]



    # 生成波浪形坐标映射
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    x_map = x_coords + wave_amp * np.sin(2 * np.pi * wave_freq * y_coords)
    y_map = y_coords + wave_amp * np.sin(2 * np.pi * wave_freq * x_coords)
    x_map = x_map.astype(np.float32)
    y_map = y_map.astype(np.float32)

    # 使用 remap 重映射图像
    distorted = cv2.remap(image, x_map, y_map, interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)  # BORDER_REFLECT 减少黑边

    return distorted


def apply_pinch(image, center, strength, radius):
    h, w = image.shape[:2]
    x_map, y_map = np.meshgrid(np.arange(w), np.arange(h))
    x_center, y_center = center
    
    # 计算每个点到中心的距离
    dx = x_map - x_center
    dy = y_map - y_center
    distance = np.sqrt(dx*dx + dy*dy)
    
    # 对于超过半径的点不应用变换
    scale = np.where(distance <= radius, 1 + strength * (radius - distance) / radius, 1.0)
    
    # 新坐标
    new_x = x_center + scale * dx
    new_y = y_center + scale * dy
    
    # 应用 remap
    distorted = cv2.remap(image, new_x.astype(np.float32), new_y.astype(np.float32),
                          interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return distorted

# 主程序
if __name__ == "__main__":
    # 读取图像
    img = cv2.imread('pano004220.jpg')  # 替换为你的图片路径
    if img is None:
        print("❌ 图像加载失败，请检查路径")
    else:
        print("✅ 图像加载成功")

        # 应用扭曲效果
        #result = apply_distortion(img, grid_size=30, max_shift=15)
        #result = apply_distortion_fast(img)
        #result = radial_squeeze(img)
        result = apply_wavy_squeeze_fast(img)
        #result = dist(result) 
        
        #result = apply_pinch(img, center=(img.shape[1]//2, img.shape[0]//2), strength=-0.8, radius=200)
        
        # 显示结果
        cv2.imshow('Original', img)
        cv2.imshow('Distorted', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 保存结果
        cv2.imwrite('output_distorted2.jpg', result)
        print("✅ 扭曲图像已保存为 output_distorted.jpg")