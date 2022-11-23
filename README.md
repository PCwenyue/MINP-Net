# A Robust Infrared Small Target Detection Method Jointing Multiple Information and Noise Prediction：Algorithm and Benchmark
The article has been submitted to IEEE Transactions on Geoscience and Remote Sending

## Algorithm Introduction
we propose in this paper a robust infrared small target detection method jointing multiple information and noise prediction, named MINP-Net.


## Dataset Introduction
We contribute an infrared small target segmentation dataset, called NCHU-Seg. The presented NCHU-Seg dataset consists of 590 infrared images which are almost selected from the real-world infrared images photographed. The infrared small targets mainly include aircraft, birds and ships, and the target sizes are distributed between 3×3 pixels and 9×9 pixels within a 256×256 infrared image, which are strictly meet the SPIE definition for infrared small target.


## Prerequisite
* Tested on Ubuntu 20.04, with Python 3.9, PyTorch 1.11, Torchvision 0.12.0, CUDA 11.3.1, and 1x NVIDIA 2080Ti 
* [The NUAA-SIRST download dir](https://github.com/YimianDai/sirst) [[ACM]](https://arxiv.org/pdf/2009.14530.pdf)


## Usage
#### 1. Train.

```bash
python train.py
```

#### 2. Test.

```bash
python test.py 
```

#### (Optional 1) Visulize your predicts.

```bash
python visulization.py
```

## Referrences
1. Li, Boyang and Xiao, Chao and Wang, Longguang et al., Dense Nested Attention Network for Infrared Small Target Detection.//arXiv preprint arXiv:2106.00487. 2021. [[code]](https://github.com/Lliu666/DNANet_BatchFormer) 



