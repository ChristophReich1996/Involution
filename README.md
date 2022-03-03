# Involution: Inverting the Inherence of Convolution for Visual Recognition
Unofficial **PyTorch** reimplementation of the paper [Involution: Inverting the Inherence of Convolution for Visual Recognition](https://arxiv.org/pdf/2103.06255.pdf)
by Duo Li, Jie Hu, Changhu Wang et al. published at CVPR 2021.

**This repository includes a pure PyTorch implementation of a 2D and 3D involution.**

Please note that the [official implementation](https://github.com/d-li14/involution) provides a more memory efficient
CuPy implementation of the 2D involution. Additionally, [shikishima-TasakiLab](https://github.com/shikishima-TasakiLab) provides a fast and memory efficent [CUDA implementation](https://github.com/shikishima-TasakiLab/Involution-PyTorch) of the 2D Involution.

## Installation
The 2D and 3D involution can be easily installed by using `pip`.
````shell script
pip install git+https://github.com/ChristophReich1996/Involution
````

## Example Usage
Additional examples, such as strided involutions or transposed convolution like involutions, can be found in the 
[example.py](examples.py) file.

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
| sigma_mapping | Non-linear mapping as introduced in the paper. If none BN + ReLU is utilized (default=None) | Optional[nn.Module] |
| kernel_size | Kernel size to be used (default=(7, 7)) | Union[int, Tuple[int, int]] |
| stride | Stride factor to be utilized (default=(1, 1)) | Union[int, Tuple[int, int]] |
| groups | Number of groups to be employed (default=1) | int |
| reduce_ratio | Reduce ration of involution channels (default=1) | int |
| dilation | Dilation in unfold to be employed (default=(1, 1)) | Union[int, Tuple[int, int]] |
| padding | Padding to be used in unfold operation (default=(3, 3)) | Union[int, Tuple[int, int]] |
| bias | If true bias is utilized in each convolution layer (default=False) | bool |
| force_shape_match | If true potential shape mismatch is solved by performing avg pool (default=False) | bool |
| **kwargs | Unused additional key word arguments | Any |

The 3D involution can be used as a `nn.Module` as follows:
````python
import torch
from involution import Involution3d

involution = Involution3d(in_channels=8, out_channels=16)
output = involution(torch.rand(1, 8, 32, 32, 32))
````

The 3D involution takes the following parameters.

| Parameter | Description | Type |
| ------------- | ------------- | ------------- |
| in_channels | Number of input channels | int |
| out_channels | Number of output channels | int |
| sigma_mapping | Non-linear mapping as introduced in the paper. If none BN + ReLU is utilized | Optional[nn.Module] |
| kernel_size | Kernel size to be used (default=(7, 7, 7)) | Union[int, Tuple[int, int, int]] |
| stride | Stride factor to be utilized (default=(1, 1, 1)) | Union[int, Tuple[int, int, int]] |
| groups | Number of groups to be employed (default=1) | int |
| reduce_ratio | Reduce ration of involution channels (default=1) | int |
| dilation | Dilation in unfold to be employed (default=(1, 1, 1)) | Union[int, Tuple[int, int, int]] |
| padding | Padding to be used in unfold operation (default=(3, 3, 3)) | Union[int, Tuple[int, int, int]] |
| bias | If true bias is utilized in each convolution layer (default=False) | bool |
| force_shape_match | If true potential shape mismatch is solved by performing avg pool (default=False) | bool |
| **kwargs | Unused additional key word arguments | Any |


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
