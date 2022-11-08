# [ICIP 2022] [Half Wavelet Attention on M-Net+ for Low-light Image Enhancement](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9897503)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/half-wavelet-attention-on-m-net-for-low-light/low-light-image-enhancement-on-lol)](https://paperswithcode.com/sota/low-light-image-enhancement-on-lol?p=half-wavelet-attention-on-m-net-for-low-light)  

## [Chi-Mao Fan](https://github.com/FanChiMao), Tsung-Jung Liu, Kuan-Hsien Liu   
[![paper](https://img.shields.io/badge/arXiv-Paper-brightgreen)](https://arxiv.org/abs/2203.01296) 
[![official_paper](https://img.shields.io/badge/IEEE-Paper-blue)](https://ieeexplore.ieee.org/document/9897503) 
[![video](https://img.shields.io/badge/Video-Presentation-F9D371)](https://youtu.be/docLJbxJkyA) 
[![slides](https://img.shields.io/badge/Presentation-Slides-B762C1)](https://docs.google.com/presentation/d/1S9NA2iRXbMGQEcEupBIzcomO-fi4y0i7/edit?usp=sharing&ouid=108348190349543369603&rtpof=true&sd=true) 
[![poster](https://img.shields.io/badge/Summary-Poster-9cf)](https://drive.google.com/file/d/1TWtzoS5TusoTgXhohxd0gmqKybfg2jnk/view?usp=sharing) 
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/52Hz/HWMNet_lowlight_enhancement)  

> Abstract : Low-Light Image Enhancement is a compute vision task which reinforces
the dark images to appropriate brightness. It can also be
seen as an ill-posed problem in image restoration domain. With
the success of deep neural networks, the convolutional neural networks
surpass the traditional algorithm-based methods and become
the mainstream in the computer vision area. To advance the performance
of enhancement algorithms, we propose an image enhancement
network (HWMNet) basing on an improved hierarchical architecture
model: M-Net+. Specifically, we use a half wavelet attention
block on M-Net+ to enrich the features of wavelet domain. Furthermore,
our HWMNet has competitive performance results on two
image enhancement datasets in terms of quantitative metrics and visual
quality.  

## Network Architecture  
<table>
  <tr>
    <td colspan="2"><img src = "https://i.imgur.com/BeT2g0y.png" alt="HWMNet" width="800"> </td>  
  </tr>
  <tr>
    <td colspan="2"><p align="center"><b>Overall Framework of HWMNet</b></p></td>
  </tr>

  <tr>
    <td> <img src = "https://i.imgur.com/FPR0WoU.png" width="400"> </td>
    <td> <img src = "https://i.imgur.com/xMVL6N1.png" width="400"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>Half Wavelet Attention Block (HWAB)</b></p></td>
    <td><p align="center"> <b>Resizing Block (Pixel Shuffle)</b></p></td>
  </tr>
</table>

## Quick Run  
You can simply demo on my space of **[Hugging Face](https://huggingface.co/spaces/52Hz/HWMNet_lowlight_enhancement)**  

or test on local environment:  

To test the pre-trained models of enhancing on your own images, run
```
python demo.py --input_dir images_folder_path --result_dir save_images_here --weights path_to_models
```
**All pre-trained models can be downloaded at [pretrained_model/README.md](pretrained_model/README.md) or [here](https://github.com/FanChiMao/HWMNet/releases)**  

## Train  
To train the restoration models of low-light enhancement. You should check the following components are correct:  
- `training.yaml`:  
  ```
  # Training configuration
  GPU: [0,1,2,3]

  VERBOSE: False

  MODEL:
    MODE: 'HWMNet-96-LOL'

  # Optimization arguments.
  OPTIM:
    BATCH: 2
    EPOCHS: 100
    # EPOCH_DECAY: [10]
    LR_INITIAL: 2e-4
    LR_MIN: 1e-6
    # BETA1: 0.9

  TRAINING:
    VAL_AFTER_EVERY: 1
    RESUME: False
    TRAIN_PS: 256
    VAL_PS: 256
    TRAIN_DIR: './datasets/LOL/train'       # path to training data
    VAL_DIR: './datasets/LOL/test' # path to validation data
    SAVE_DIR: './checkpoints'           # path to save models and images
  ```
  
- Dataset:  
  The preparation of dataset in more detail, see [datasets/README.md](datasets/README.md).  
  
- Train:  
  If the above path and data are all correctly setting, just simply run:  
  ```
  python train.py
  ```  

- Details of Hyperparameters
	```
	-------------------------------------------------
	LOL dataset:
	Training patches: 4850 (485 x 10)
	Validation: 15
	Initial learning rate: 5e-5
	Final learning rate: 1e-5
	Training epochs: 300 (200 is enough)

	MIT-5K dataset:
	Training patches: 36000 (4500 x 8)
	Validation: 500
	Initial learning rate: 1e-4
	Final learning rate: 1e-6
	Training epochs: 100
	-------------------------------------------------
	```

## Test (Evaluation)  
 
- To test the PSNR, SSIM and LPIPS of *image enhancement*, see [evaluation.py](./evaluation.py) and run
```
python evaluation.py -dirA images_folder_path -dirB images_folder_path -type image_data_type --use_gpu use_gpu_or_not
```

## Result  
- Evaluation on LOL dataset  
<img src = "https://i.imgur.com/NUIAWaN.png" width="450">  

- Evaluation on MIT-Adobe FiveK dataset  
<img src = "https://i.imgur.com/Dw6vhmq.png" width="800">  

## Visual Comparison  
- Evaluation on LOL dataset  
<img src = "https://i.imgur.com/qmxlPOG.png" width="800">  

- Evaluation on MIT-Adobe FiveK dataset  
<img src = "https://i.imgur.com/EIbfRzx.png" width="500">  

**More visual results can be downloaded at [here](https://github.com/FanChiMao/HWMNet/releases).**  


## Citation  

```
@article{fan2022half,
  title={Half Wavelet Attention on M-Net+ for Low-Light Image Enhancement},
  author={Fan, Chi-Mao and Liu, Tsung-Jung and Liu, Kuan-Hsien},
  journal={arXiv preprint arXiv:2203.01296},
  year={2022}
}
```
## Contact  
If you have any question, feel free to contact qaz5517359@gmail.com  

![visitors](https://visitor-badge.glitch.me/badge?page_id=HWMNet_github)  
