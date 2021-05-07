# Involution: Inverting the Inherence of Convolution for Visual Recognition
Unofficial **PyTorch** reimplementation of the paper [Involution: Inverting the Inherence of Convolution for Visual Recognition](https://arxiv.org/pdf/2103.06255.pdf)
by Duo Li, Jie Hu, Changhu Wang et al. published at CVPR 2021.

Please note that the [official implementation](https://github.com/d-li14/involution) provides a more memory efficient
CuPy implementation of the 2d involution.

## Example Usage
The 2D involution can be used as a `nn.Module` as follows:
````python
import torch
from involution import Involution2d

involution = Involution2d(in_channels=32, out_channels=64)
output = involution(torch.rand(1, 32, 128, 128))
````

The 2D involution takes the following parameters.

| Parameter | Description | Type |
| ------------- | ------------- | ------------- |
| in_channels | Number of input channels | int |
| out_channels | Number of output channels | int |
| sigma_mapping | Non-linear mapping as introduced in the paper. If none BN + ReLU is utilized | nn.Module |
| kernel_size | Kernel size to be used | Union[int, Tuple[int, int]] |
| stride | Stride factor to be utilized | Union[int, Tuple[int, int]] |
| groups | Number of groups to be employed | int |
| reduce_ratio | Reduce ration of involution channels | int |
| dilation | Dilation in unfold to be employed | Union[int, Tuple[int, int]] |
| padding | Padding to be used in unfold operation | Union[int, Tuple[int, int]] |
| bias | If true bias is utilized in each convolution layer | bool |
| **kwargs | Unused additional key word arguments | Any |

## Installation
The 2D involution can be easily installed by utilizing `pip`.
````shell script
pip install git+https://github.com/ChristophReich1996/Involution
````

## Reference

````bibtex
@inproceedings{Li2021,
    author = {Li, Duo and Hu, Jie and Wang, Changhu and Li, Xiangtai and She, Qi and Zhu, Lei and Zhang, Tong and Chen, Qifeng},
    title = {Involution: Inverting the Inherence of Convolution for Visual Recognition},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2021}
}
````
