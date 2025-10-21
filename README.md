# Enhancing Parking-slot Detection with Efficient Synthetic Data Generation




## 1. Enviroment setup

### 1.1 Image inpainting model.
 We use LaMa for image inpainting. As we don't want to be bothered by different pytorch versions, we recommand using modelscope.
 pip install modelscope==1.13.0

 The torch version we use is torch 2.3.1+cu118, torchaudio 2.3.1+cu118, torchvision 0.18.1+cu118

### 1.2 Parklot augmentation
 pip install opencv-python==4.12.0.88
   

## 2. Image process pipeline

### 2.1 Remove original parklot by image inpainting.
 In image_inpainting.py, 
 ```
    input_location = './data/pano004220.jpg'
    input_mask_location_ori = './data/pano004220.png'
    output_location = input_location.replace('.','oimage.')
    inpaint_image(input_location, input_mask_location_ori, output_location)
 #The code was mofifyed from modelscope https://www.modelscope.cn/models/iic/cv_fft_inpainting_lama/summary.
```
### 2.2 Parklot augmentation.
The avm images we use was got from ps2.0 in https://cslinzhang.github.io/deepps/.
In modify_parklot.py, it takes inpainted avm image and original mask from image_inpating.py, then produce new parklot.


### 2.3 Parklot generation.
The avm images we use was got from ps2.0 in https://cslinzhang.github.io/deepps/.
In modify_parklot.py, it takes inpainted avm image and original mask from image_inpating.py, then produce new parklot.


## 3. Ref:

@article{suvorov2021resolution,
  title={Resolution-robust Large Mask Inpainting with Fourier Convolutions},
  author={Suvorov, Roman and Logacheva, Elizaveta and Mashikhin, Anton and Remizova, Anastasia and Ashukha, Arsenii and Silvestrov, Aleksei and Kong, Naejin and Goka, Harshith and Park, Kiwoong and Lempitsky, Victor},
  journal={arXiv preprint arXiv:2109.07161},
  year={2021}
}

@article{kulshreshtha2022feature,
  title={Feature Refinement to Improve High Resolution Image Inpainting},
  author={Kulshreshtha, Prakhar and Pugh, Brian and Jiddi, Salma},
  journal={arXiv preprint arXiv:2206.13644},
  year={2022}
}

@article{8412601,
  author={Zhang, Lin and Huang, Junhao and Li, Xiyuan and Xiong, Lu},
  journal={IEEE Transactions on Image Processing}, 
  title={Vision-Based Parking-Slot Detection: A DCNN-Based Approach and a Large-Scale Benchmark Dataset}, 
  year={2018},
  volume={27},
  number={11},
  pages={5350-5364},
  keywords={Feature extraction;Cameras;Convolutional neural networks;Detectors;Transforms;Space vehicles;Self-parking systems;parking-slot detection;deep convolutional neural networks},
  doi={10.1109/TIP.2018.2857407}}
