import cv2
import numpy as np
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

#input_location = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_inpainting/image_inpainting.png'
#input_mask_location = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_inpainting/image_inpainting_mask.png'


def pre_process_mask_psv2(mask):
    mask[mask == 3] == 0   #do not modify road
    mask = mask *255
    kernel = np.ones((4, 4), np.uint8)
    mask_dilate = cv2.dilate(mask, kernel, iterations=6).astype(np.uint8)

    return mask_dilate

def pre_proce_file_psv2(mask_path):
    mask = cv2.imread(mask_path)
    input_mask_locaion = mask_path.replace('.','inmask.')
    mask_out = pre_process_mask_psv2(mask)
    cv2.imwrite(input_mask_locaion,mask_out)
    return input_mask_locaion



def inpaint_image(input_image_path, input_mask_path, out_image_path):
     input_mask_location = pre_proce_file_psv2(input_mask_location_ori)
     
     input = {
        'img':input_location,
        'mask':input_mask_location,
        }
     image = cv2.imread(input['img'])
     image_ori_shape = image.shape[:2]
     inpainting = pipeline(Tasks.image_inpainting, model='iic/cv_fft_inpainting_lama', refine=True)
     result = inpainting(input)
     vis_img = result[OutputKeys.OUTPUT_IMG]
     out_img = cv2.resize(vis_img,image_ori_shape)
     #cv2.imwrite('result_6.png', vis_img)
     cv2.imwrite(out_image_path, out_img)
     

if __name__ == "__main__":
    input_location = './data/pano004220.jpg'
    input_mask_location_ori = './data/pano004220.png'
    output_location = input_location.replace('.','oimage.')
    inpaint_image(input_location, input_mask_location_ori, output_location)