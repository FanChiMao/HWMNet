# Half Wavelet Attention on M-Net+ for Low-light Image Enhancement  

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
    <td colspan="2"><img src = "https://i.imgur.com/GYeypta.png" alt="HWMNet" width="800"> </td>  
  </tr>
  <tr>
    <td colspan="2"><p align="center"><b>Overall Framework of SRMNet</b></p></td>
  </tr>
  
  <tr>
    <td> <img src = "https://i.imgur.com/z6Vds87.png" width="400"> </td>
    <td> <img src = "https://i.imgur.com/eaLejBK.png" width="400"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>Selective Residual Block (SRB)</b></p></td>
    <td><p align="center"> <b>Resizing Block (Pixel Shuffle)</b></p></td>
  </tr>
</table>


## Quick Run  
You can simply demo on the space of [Hugging Face](https://huggingface.co/spaces/52Hz/HWMNet_lowlight_enhancement):  

Or test on local environment:  

To test the pre-trained models of Denoising on your own images, run
```
python demo.py --input_dir images_folder_path --result_dir save_images_here --weights path_to_models
```
**All pre-trained models can be downloaded at [pretrained_model/README.md](pretrained_model/README.md) or [here](https://github.com/FanChiMao/SRMNet/releases)**  

## Train  
To train the restoration models of Denoising. You should check the following components are correct:  
- `training.yaml`:  
  ```
  # Training configuration
  GPU: [0,1,2,3]

  VERBOSE: False

  MODEL:
    MODE: 'SRMNet_denoise'

  # Optimization arguments.
  OPTIM:
    BATCH: 2
    EPOCHS: 100
    # EPOCH_DECAY: [10]
    LR_INITIAL: 1e-4
    LR_MIN: 1e-6
    # BETA1: 0.9

  TRAINING:
    VAL_AFTER_EVERY: 1
    RESUME: False
    TRAIN_PS: 256
    VAL_PS: 256
    TRAIN_DIR: 'D:/PycharmProjects/SUNet-main/datasets/Denoising_DIV2K/train'       # path to training data
    VAL_DIR: 'D:/PycharmProjects/SUNet-main/datasets/Denoising_DIV2K/test' # path to validation data
    SAVE_DIR: './checkpoints'           # path to save models and images
  ```
  
- Dataset:  
  The preparation of dataset in more detail, see [Dataset/README.md](Dataset/README.md).  
  
- Train:  
  If the above path and data are all correctly setting, just simply run:  
  ```
  python train.py
  ```  
## Test (Evaluation)  

- To run the models of real image denoising, see [test_DND_real_denoising.py](./test_DND_real_denoising.py) and [test_SIDD_real_denoising.py](./test_SIDD_real_denoising.py).  
- To test the PSNR and SSIM of *real image denoising*, see [evaluation_DND.py](./evaluation_DND.py) and [evaluation_SIDD.m](./evaluation_SIDD.m).  
- To test the PSNR and SSIM of *AWGN image denoising*, see the [evaluation.m](./evaluation.m).  

## Result  
- AWGN image denoising  
<img src = "https://i.imgur.com/TILnGHa.png" width="800">  

- Real image denoising  
<img src = "https://i.imgur.com/vxt6Vs9.png" width="400">  

## Visual Comparison  

<img src = "https://i.imgur.com/H9CWlll.png" width="800">  

**More visual results can be downloaded at [here](https://github.com/FanChiMao/SRMNet/releases).**  


## Citation  

## Contact

