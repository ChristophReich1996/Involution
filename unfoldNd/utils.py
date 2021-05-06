"""Shared utility functions."""

import numpy
from torch.nn.functional import conv1d, conv2d, conv3d
from torch.nn.modules.utils import _pair, _single, _triple


def _get_kernel_size_numel(kernel_size):
    """Determine number of pixels/voxels. ``kernel_size`` must be an ``N``-tuple."""
    if not isinstance(kernel_size, tuple):
        raise ValueError("kernel_size must be a tuple. Got {}.".format(kernel_size))

    return _get_numel_from_shape(kernel_size)


def _get_numel_from_shape(shape_tuple):
    """Compute number of elements from shape."""
    return int(numpy.prod(shape_tuple))


def _tuple(kernel_size, N):
    """Turn ``kernel_size`` argument of ``N``d convolution into an ``N``-tuple."""
    if N == 1:
        return _single(kernel_size)
    elif N == 2:
        return _pair(kernel_size)
    elif N == 3:
        return _triple(kernel_size)
    else:
        _raise_dimension_error(N)


def _get_conv(N):
    """Return convolution operation used to perform unfolding."""
    if N == 1:
        return conv1d
    elif N == 2:
        return conv2d
    if N == 3:
        return conv3d
    else:
        _raise_dimension_error(N)


def _raise_dimension_error(N):
    """Notify user that inferred input dimension is not supported."""
    raise ValueError("Only 1,2,3-dimensional unfold is supported. Got N={}.".format(N))
