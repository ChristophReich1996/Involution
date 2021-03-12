from typing import Union, Tuple

import torch
import torch.nn as nn


class Involution2d(nn.Module):
    """
    This class implements the 2d involution proposed in:
    https://arxiv.org/pdf/2103.06255.pdf
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]] = (7, 7),
                 stride: Union[int, Tuple[int, int]] = (1, 1),
                 groups: int = 1,
                 reduce_ratio: int = 4,
                 dilation: Union[int, Tuple[int, int]] = (1, 1),
                 padding: Union[int, Tuple[int, int]] = (3, 3)) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param kernel_size: (Union[int, Tuple[int, int]]) Kernel size to be used
        :param stride: (Union[int, Tuple[int, int]]) Stride factor to be utilized
        :param groups: (int) Number of groups to be employed
        :param reduce_ratio: (int) Reduce ration of involution channels
        :param dilation: (Union[int, Tuple[int, int]]) Dilation in unfold to be employed
        :param padding: (Union[int, Tuple[int, int]]) Padding to be used in unfold operation
        """
        # Call super constructor
        super(Involution2d, self).__init__()
        # Save parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else tuple(kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else tuple(stride)
        self.groups = groups
        self.reduce_ratio = reduce_ratio
        self.dilation = dilation if isinstance(dilation, tuple) else tuple(dilation, dilation)
        self.padding = padding if isinstance(padding, tuple) else tuple(padding, padding)
        # Init modules
        self.initial_mapping = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                             kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                             bias=False) if self.in_channels != self.out_channels else nn.Identity()
        self.o_mapping = nn.AvgPool2d(kernel_size=self.stride, stride=self.stride)
        self.reduce_mapping = nn.Conv2d(in_channels=self.in_channels,
                                        out_channels=self.out_channels // self.reduce_ratio, kernel_size=(1, 1),
                                        stride=(1, 1), padding=(0, 0), bias=False)
        self.span_mapping = nn.Conv2d(in_channels=self.out_channels // self.reduce_ratio,
                                      out_channels=self.kernel_size[0] * self.kernel_size[1] * self.groups,
                                      kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, dilation=dilation, padding=padding, stride=stride)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of the shape []
        :return: (torch.Tensor) Output tensor of the shape []
        """
        # Check input dimension of input tensor
        assert input.ndimension() == 4, \
            "Input tensor to involution must be 4d but {}d tensor is given".format(input.ndimension())
        # Save input shape
        batch_size, in_channels, height, width = input.shape
        # Unfold and reshape input tensor
        input_unfolded = self.unfold(self.initial_mapping(input))
        input_unfolded = input_unfolded.view(batch_size, self.groups, self.out_channels // self.groups,
                                             self.kernel_size[0] * self.kernel_size[1], height, width)
        # Generate kernel
        kernel = self.span_mapping(self.reduce_mapping(self.o_mapping(input)))
        kernel = kernel.view(
            batch_size, self.groups, self.kernel_size[0] * self.kernel_size[1], height, width).unsqueeze(dim=2)
        # Apply kernel to produce output
        output = (kernel * input_unfolded).sum(dim=3).view(batch_size, -1, height, width)
        return output
