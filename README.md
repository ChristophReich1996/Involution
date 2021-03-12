# Involution: Inverting the Inherence of Convolution for Visual Recognition
Unofficial **PyTorch** reimplementation of the paper [Involution: Inverting the Inherence of Convolution for Visual Recognition](https://arxiv.org/pdf/2103.06255.pdf)
[CRPR2021] by Duo Li, Jie Hu, Changhu Wang et al.

Example usage of the 2d involution
````python
import torch
from involution import Involution2d

involution = Involution2d(in_channels=32, out_channels=64, groups=16)
output = involution(torch.rand(1, 32, 128, 128))
````