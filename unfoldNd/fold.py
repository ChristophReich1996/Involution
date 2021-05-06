"""Generalization of fold operation."""

import torch

from unfoldNd.unfold import unfoldNd
from unfoldNd.utils import _get_kernel_size_numel, _get_numel_from_shape, _tuple


class FoldNd(torch.nn.Module):
    """Combines an array of sliding local blocks into a large containing tensor.

    Also known as col2im.

    PyTorch module that accepts 3d, 4d, and 5d tensors. Acts like ``torch.nn.Fold``
    for a 4d input. Unfolds an index tensor to scatter sliding blocks to the result.

    See docs at https://pytorch.org/docs/stable/generated/torch.nn.Fold.html.
    """

    def __init__(self, output_size, kernel_size, dilation=1, padding=0, stride=1):
        super().__init__()

        self._output_size = output_size
        self._kernel_size = kernel_size
        self._dilation = dilation
        self._padding = padding
        self._stride = stride

    def forward(self, input):
        return foldNd(
            input,
            self._output_size,
            self._kernel_size,
            dilation=self._dilation,
            padding=self._padding,
            stride=self._stride,
        )


def foldNd(input, output_size, kernel_size, dilation=1, padding=0, stride=1):
    """Combines an array of sliding local blocks into a large containing tensor.

    Also known as col2im.

    Pytorch functional that accepts 3d, 4d, and 5d tensors. Acts like
    ``torch.nn.functional.fold`` for a 4d input. Unfolds an index tensor to scatter
    sliding blocks to the result.

    See docs at https://pytorch.org/docs/stable/nn.functional.html#fold.

    Raises:
        ValueError: If ``output_size`` is not specified as ``tuple``. Otherwise
            the fold dimension ``N`` cannot be inferred.
        RuntimeError: If the output has more entries than can be exactly represented
            with ``float32`` and the implementation's correctness breaks down.
    """
    device = input.device

    if isinstance(output_size, tuple):
        N = len(output_size)
        output_size_numel = _get_numel_from_shape(output_size)
    else:
        raise ValueError("'output_size' must be tuple. Got {}.".format(type(output_size)))

    kernel_size = _tuple(kernel_size, N)
    kernel_size_numel = _get_kernel_size_numel(kernel_size)

    batch_size = input.shape[0]
    in_channels_kernel_size_numel = input.shape[1]
    in_channels = in_channels_kernel_size_numel // kernel_size_numel

    _check_output_size(output_size_numel)
    idx = torch.arange(output_size_numel, dtype=torch.float32, device=device).reshape(
        1, 1, *output_size
    )
    idx = unfoldNd(idx, kernel_size, dilation=dilation, padding=padding, stride=stride)

    input = input.reshape(batch_size, in_channels, -1)
    idx = idx.reshape(1, 1, -1).long().expand(batch_size, in_channels, -1)

    output = torch.zeros(
        batch_size, in_channels, output_size_numel, device=device, dtype=input.dtype
    )
    output.scatter_add_(2, idx, input)

    return output.reshape(batch_size, in_channels, *output_size)


def _check_output_size(output_size_numel):
    """Raise exception if output size has more elements than float32 can represent."""
    # arg min long(float32(x)) != x,  see https://stackoverflow.com/q/27207149
    num_bits_mantissa = 23
    min_int_inexact_as_float32 = 2 ** (num_bits_mantissa + 1) + 1

    if output_size_numel >= min_int_inexact_as_float32:
        raise RuntimeError(
            "Output size elements({}) must be smaller than {}".format(output_size_numel,min_int_inexact_as_float32)
        )
