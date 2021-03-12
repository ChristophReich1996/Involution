# Involution: Inverting the Inherence of Convolution for Visual Recognition
Unofficial **PyTorch** reimplementation of the paper [Involution: Inverting the Inherence of Convolution for Visual Recognition](https://arxiv.org/pdf/2103.06255.pdf)
by Duo Li, Jie Hu, Changhu Wang et al. published at CVPR 2021.

Please note that the [official implementation](https://github.com/d-li14/involution) provides a more memory efficient
CuPy implementation of the 2d involution.

## Example usage
The 2d involution can be used as a `nn.Module` as follows:
````python
import torch
from involution import Involution2d

involution = Involution2d(in_channels=32, out_channels=64)
output = involution(torch.rand(1, 32, 128, 128))
````

## Installation
The 2d involution can be easily installed by utilizing `pip`
````shell script
pip install git+https://github.com/ChristophReich1996/Involution
````