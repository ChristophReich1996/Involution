import torch
from involution import Involution2d, Involution3d

if __name__ == '__main__':
    # 2D involution example
    involution_2d = Involution2d(in_channels=4, out_channels=8)
    input = torch.rand(2, 4, 64, 64)
    output = involution_2d(input)

    # 2D involution as transposed convolution
    involution_2d = Involution2d(in_channels=6, out_channels=12)
    input_ = torch.rand(2, 6, 4, 4)
    input = torch.zeros(2, 6, 8, 8)
    input[..., ::2, ::2] = input_
    output = involution_2d(input)

    # 2D involution with stride
    involution_2d = Involution2d(in_channels=4, out_channels=8, stride=2, kernel_size=2, padding=0)
    input = torch.rand(2, 4, 32, 32)
    output = involution_2d(input)

    # 3D involution example
    involution_3d = Involution3d(in_channels=8, out_channels=16)
    input = torch.rand(1, 8, 32, 32, 32)
    output = involution_3d(input)

    # 3D involution with stride
    involution_3d = Involution3d(in_channels=8, out_channels=16, stride=2, kernel_size=2, padding=0)
    input = torch.rand(1, 8, 16, 16, 16)
    output = involution_3d(input)