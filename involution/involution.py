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
                 sigma_mapping: nn.Module = None,
                 kernel_size: Union[int, Tuple[int, int]] = (7, 7),
                 stride: Union[int, Tuple[int, int]] = (1, 1),
                 groups: int = 1,
                 reduce_ratio: int = 1,
                 dilation: Union[int, Tuple[int, int]] = (1, 1),
                 padding: Union[int, Tuple[int, int]] = (3, 3),
                 **kwargs) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param sigma_mapping: (nn.Module) Non-linear mapping as introduced in the paper. If none BN + ReLU is utilized
        :param kernel_size: (Union[int, Tuple[int, int]]) Kernel size to be used
        :param stride: (Union[int, Tuple[int, int]]) Stride factor to be utilized
        :param groups: (int) Number of groups to be employed
        :param reduce_ratio: (int) Reduce ration of involution channels
        :param dilation: (Union[int, Tuple[int, int]]) Dilation in unfold to be employed
        :param padding: (Union[int, Tuple[int, int]]) Padding to be used in unfold operation
        :param **kwargs: Unused additional key word arguments
        """
        # Call super constructor
        super(Involution2d, self).__init__()
        # Check parameters
        assert isinstance(in_channels, int) and in_channels > 0, "in channels must be a positive integer."
        assert in_channels % groups == 0, "out_channels must be divisible by groups"
        assert isinstance(out_channels, int) and out_channels > 0, "out channels must be a positive integer."
        assert out_channels % groups == 0, "out_channels must be divisible by groups"
        assert isinstance(sigma_mapping, nn.Module) or sigma_mapping is None, \
            "Sigma mapping must be an nn.Module or None to utilize the default mapping (BN + ReLU)."
        assert isinstance(kernel_size, int) or isinstance(kernel_size, tuple), \
            "kernel size must be an int or a tuple of ints."
        assert isinstance(stride, int) or isinstance(stride, tuple), \
            "stride must be an int or a tuple of ints."
        assert isinstance(groups, int), "groups must be a positive integer."
        assert isinstance(reduce_ratio, int) and reduce_ratio > 0, "reduce ratio must be a positive integer."
        assert isinstance(dilation, int) or isinstance(dilation, tuple), \
            "dilation must be an int or a tuple of ints."
        assert isinstance(padding, int) or isinstance(padding, tuple), \
            "padding must be an int or a tuple of ints."
        # Save parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else tuple(kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else tuple(stride, stride)
        self.groups = groups
        self.reduce_ratio = reduce_ratio
        self.dilation = dilation if isinstance(dilation, tuple) else tuple(dilation, dilation)
        self.padding = padding if isinstance(padding, tuple) else tuple(padding, padding)
        # Init modules
        self.sigma_mapping = sigma_mapping if sigma_mapping is not None else nn.Sequential(
            nn.BatchNorm2d(num_features=self.out_channels // self.reduce_ratio, momentum=0.3), nn.ReLU())
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

    def __repr__(self) -> str:
        """
        Method returns information about the module
        :return: (str) Info string
        """
        return ("{}({}, {}, kernel_size=({}, {}), stride=({}, {}), padding=({}, {}), "
                "groups={}, reduce_ratio={}, dilation=({}, {}), sigma_mapping={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.kernel_size[0],
            self.kernel_size[1],
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            self.groups,
            self.reduce_mapping,
            self.dilation[0],
            self.dilation[1],
            str(self.sigma_mapping)
        ))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of the shape [batch size, in channels, height, width]
        :return: (torch.Tensor) Output tensor of the shape [batch size, out channels, height, width] (w/ same padding)
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
        kernel = self.span_mapping(self.sigma_mapping(self.reduce_mapping(self.o_mapping(input))))
        kernel = kernel.view(
            batch_size, self.groups, self.kernel_size[0] * self.kernel_size[1], height, width).unsqueeze(dim=2)
        # Apply kernel to produce output
        output = (kernel * input_unfolded).sum(dim=3).view(batch_size, -1, height, width)
        return output
        

class Unfold3d(torch.nn.Module):
    """Extracts sliding local blocks from a batched input tensor. Also known as im2col.

    PyTorch module that accepts 3d, 4d, and 5d tensors. Acts like ``torch.nn.Unfold``
    for a 4d input. Uses one-hot convolution under the hood.

    See docs at https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html.
    """

    def __init__(self, kernel_size, dilation=1, padding=0, stride=1):
        super().__init__()

        self._kernel_size = kernel_size
        self._dilation = dilation
        self._padding = padding
        self._stride = stride

    def forward(self, input):
        return self.unfold3d(
            input,
            self._kernel_size,
            dilation=self._dilation,
            padding=self._padding,
            stride=self._stride,
        )

    def unfold3d(self, input, kernel_size, dilation=1, padding=0, stride=1):
        """Extracts sliding local blocks from a batched input tensor. Also known as im2col.

        Pytorch functional that accepts 3d, 4d, and 5d tensors. Acts like
        ``torch.nn.functional.unfold`` for a 4d input. Uses one-hot convolution under the
        hood.

        See docs at https://pytorch.org/docs/stable/nn.functional.html#unfold.
        """
        batch_size, in_channels = input.shape[0], input.shape[1]

        # prepare one-hot convolution kernel
        kernel_size = torch.nn.modules.utils._triple(kernel_size)
        kernel_size_numel = int(kernel_size[0]*kernel_size[1]*kernel_size[2])
        weight = self._make_weight(in_channels, kernel_size, input.device, input.dtype)

        unfold = torch.nn.functional.conv3d(
            input,
            weight,
            bias=None,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
        )

        return unfold.reshape(batch_size, in_channels * kernel_size_numel, -1)

    def _make_weight(self, in_channels, kernel_size, device, dtype):
        """Create one-hot convolution kernel. ``kernel_size`` must be an ``N``-tuple.

        Details:
            Let ``T`` denote the one-hot weight, then
            ``T[c * i, 0, j] = δᵢⱼ ∀ c = 1, ... C_in``
            (``j`` is a group index of the ``Kᵢ``).

            This can be done by building diagonals ``D[i, j] = δᵢⱼ``, reshaping
            them into ``[∏ᵢ Kᵢ, 1, K]``, and repeat them ``C_in`` times along the
            leading dimension.

        Returns:
            torch.Tensor : A tensor of shape ``[ C_in * ∏ᵢ Kᵢ, 1, K]`` where
                ``K = (K₁, K₂, ..., Kₙ)`` is the kernel size. Filter groups are
                one-hot such that they effectively extract one element of the patch
                the kernel currently overlaps with.


        """
        kernel_size_numel = int(kernel_size[0]*kernel_size[1]*kernel_size[2])
        repeat = [in_channels, 1] + [1 for _ in kernel_size]

        return (
            torch.eye(kernel_size_numel, device=device, dtype=dtype)
            .reshape((kernel_size_numel, 1, *kernel_size))
            .repeat(*repeat)
        )


class Involution3d(nn.Module):
    """
    This class implements the 3d involution. Thanks for the code 
    from https://github.com/f-dangel/unfoldNd, which makes this 3d version become possible
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 sigma_mapping: nn.Module = None,
                 kernel_size: Union[int, Tuple[int, int]] = (7, 7, 7),
                 stride: Union[int, Tuple[int, int]] = (1, 1, 1),
                 groups: int = 1,
                 reduce_ratio: int = 1,
                 dilation: Union[int, Tuple[int, int]] = (1, 1, 1),
                 padding: Union[int, Tuple[int, int]] = (3, 3, 3),
                 **kwargs) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param sigma_mapping: (nn.Module) Non-linear mapping as introduced in the paper. If none BN + ReLU is utilized
        :param kernel_size: (Union[int, Tuple[int, int]]) Kernel size to be used
        :param stride: (Union[int, Tuple[int, int]]) Stride factor to be utilized
        :param groups: (int) Number of groups to be employed
        :param reduce_ratio: (int) Reduce ration of involution channels
        :param dilation: (Union[int, Tuple[int, int]]) Dilation in unfold to be employed
        :param padding: (Union[int, Tuple[int, int]]) Padding to be used in unfold operation
        :param **kwargs: Unused additional key word arguments
        """
        # Call super constructor
        super(Involution3d, self).__init__()
        # Check parameters
        assert isinstance(in_channels, int) and in_channels > 0, "in channels must be a positive integer."
        assert in_channels % groups == 0, "out_channels must be divisible by groups"
        assert isinstance(out_channels, int) and out_channels > 0, "out channels must be a positive integer."
        assert out_channels % groups == 0, "out_channels must be divisible by groups"
        assert isinstance(sigma_mapping, nn.Module) or sigma_mapping is None, \
            "Sigma mapping must be an nn.Module or None to utilize the default mapping (BN + ReLU)."
        assert isinstance(kernel_size, int) or isinstance(kernel_size, tuple), \
            "kernel size must be an int or a tuple of ints."
        assert isinstance(stride, int) or isinstance(stride, tuple), \
            "stride must be an int or a tuple of ints."
        assert isinstance(groups, int), "groups must be a positive integer."
        assert isinstance(reduce_ratio, int) and reduce_ratio > 0, "reduce ratio must be a positive integer."
        assert isinstance(dilation, int) or isinstance(dilation, tuple), \
            "dilation must be an int or a tuple of ints."
        assert isinstance(padding, int) or isinstance(padding, tuple), \
            "padding must be an int or a tuple of ints."
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
        self.sigma_mapping = sigma_mapping if sigma_mapping is not None else nn.Sequential(
            nn.BatchNorm3d(num_features=self.out_channels // self.reduce_ratio), nn.ReLU())
        self.initial_mapping = nn.Conv3d(in_channels=self.in_channels, out_channels=self.out_channels,
                                         kernel_size=1, stride=1, padding=0,
                                         bias=False) if self.in_channels != self.out_channels else nn.Identity()
        self.o_mapping = nn.AvgPool3d(kernel_size=self.stride, stride=self.stride)
        self.reduce_mapping = nn.Conv3d(in_channels=self.in_channels,
                                        out_channels=self.out_channels // self.reduce_ratio, kernel_size=1,
                                        stride=1, padding=0, bias=False)
        self.span_mapping = nn.Conv3d(in_channels=self.out_channels // self.reduce_ratio,
                                      out_channels=self.kernel_size[0] * self.kernel_size[1]* self.kernel_size[2] * self.groups,
                                      kernel_size=1, stride=1, padding=0)
        self.unfold = Unfold3d(kernel_size=self.kernel_size, dilation=dilation, padding=padding, stride=stride)

    def __repr__(self) -> str:
        """
        Method returns information about the module
        :return: (str) Info string
        """
        return ("{}({}, {}, kernel_size=({}, {}), stride=({}, {}), padding=({}, {}), "
                "groups={}, reduce_ratio={}, dilation=({}, {}), sigma_mapping={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.kernel_size[0],
            self.kernel_size[1],
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            self.groups,
            self.reduce_mapping,
            self.dilation[0],
            self.dilation[1],
            str(self.sigma_mapping)
        ))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of the shape [batch size, in channels, height, width]
        :return: (torch.Tensor) Output tensor of the shape [batch size, out channels, height, width] (w/ same padding)
        """
        # Check input dimension of input tensor
        assert input.ndimension() == 5, \
            "Input tensor to involution must be 5d but {}d tensor is given".format(input.ndimension())
        # Save input shape
        batch_size, in_channels, depth, height, width = input.shape
        # Unfold and reshape input tensor
        input_unfolded = self.unfold(self.initial_mapping(input))
        input_unfolded = input_unfolded.view(batch_size, self.groups, self.out_channels // self.groups,
                                             self.kernel_size[0] * self.kernel_size[1]* self.kernel_size[2],depth, height, width)
        # Generate kernel
        kernel = self.span_mapping(self.sigma_mapping(self.reduce_mapping(self.o_mapping(input))))
        kernel = kernel.view(
            batch_size, self.groups, self.kernel_size[0] * self.kernel_size[1]* self.kernel_size[2], depth, height, width).unsqueeze(dim=2)
        # Apply kernel to produce output
        output = (kernel * input_unfolded).sum(dim=3).view(batch_size, -1, depth, height, width)
        return output

