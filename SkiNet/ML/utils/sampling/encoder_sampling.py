from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Optional, Tuple, cast

from SkiNet.ML.utils.sampling.base_sampling import EncoderParams2D
from SkiNet.ML.utils.typing_utils import IntOrTuple2d, IntOrTuple2d3d, TupleOfInt2d3d, expand_to_tuple


@unique
class EncoderParamSpec(Enum):
    KERNEL = "Size of the convolving kernel"
    DILATION = "Spacing between convolving kernels"
    STRIDE = "Stride of the convolution"

@dataclass(frozen=True)
class EncoderParams:
    """
    Validated and normalized convolution parameters for encoder

    This class is mostly intended to be used via `get_encoder_params_2d` or similar factory functions
    to ensure proper validation and type safety.

    :ivar kernel: Normalized kernel size (tuple of 2 or 3 ints)
    :ivar stride: Normalized stride (tuple of 2 or 3 ints)
    :ivar dilation: Normalized dilation (tuple of 2 or 3 ints)
    All kernel, dilation and stride values are positive integer and have identical dimensionality
    :ivar padding: Automatically computed encoder-side padding based on kernel and dilation
    """
    kernel: TupleOfInt2d3d
    dilation: TupleOfInt2d3d
    stride: TupleOfInt2d3d
    padding: TupleOfInt2d3d = field(init=False)

    @classmethod
    def from_inputs(cls,
                    kernel: IntOrTuple2d3d,
                    dilation: IntOrTuple2d3d,
                    stride: IntOrTuple2d3d,
                    num_dims: Optional[int] = None) -> "EncoderParams":
        """
        Return validated and normalized convolution parameters

        :param kernel: Kernel size of the convolution operation, can be an int or a 2d or 3d tuple of int
        :param dilation: Dilation factor of the convolution operation, can be an int or a 2d or 3d tuple of int
        :param stride: Stride of the convolution operation, can be an int or a 2d or 3d tuple of int
        :param num_dims: Number of dimensions to convert to when all inputs are scalars, can be 2 or 3
        """

        if num_dims is not None and num_dims not in {2, 3}:
            raise ValueError("Invalid number of dimensions, only 2d and 3d datasets are supported")

        _validate_conv_inputs(kernel, dilation, stride)
        kernel_norm, dilation_norm = _normalize_kernel_dilation(kernel, dilation, num_dims)
        stride_norm = _normalize_stride(stride, len(kernel_norm))

        params = cls(kernel=kernel_norm, dilation=dilation_norm, stride=stride_norm)
        return params

    def __post_init__(self) -> None:
        object.__setattr__(self, "padding", get_padding(self.kernel, self.dilation))

    def as_2d(self) -> EncoderParams2D:
        """
        Return a 2D view of this EncoderParams.
        """
        _ensure_2d_params(kernel=self.kernel, stride=self.stride, dilation=self.dilation)
        return cast(EncoderParams2D, self)

def _validate_conv_inputs(kernel: IntOrTuple2d3d,
                          dilation: IntOrTuple2d3d,
                          stride: IntOrTuple2d3d) -> None:
    if isinstance(kernel, int):
        if kernel > 0 and kernel % 2 == 0:
            raise ValueError("Even kernel sizes cannot preserve spatial dimensions with symmetric 'same' padding. "
                             "Use an odd kernel size.")
    if isinstance(kernel, tuple):
        for k in kernel:
            if isinstance(k, int) and k > 0 and k % 2 == 0:
                raise ValueError("Even kernel sizes cannot preserve spatial dimensions with symmetric 'same' padding."
                                 " Use an odd kernel size.")
    kernel = _validate_param(kernel, EncoderParamSpec.KERNEL)
    dilation = _validate_param(dilation, EncoderParamSpec.DILATION)
    stride = _validate_param(stride, EncoderParamSpec.STRIDE)

def validate_conv_inputs(kernel: IntOrTuple2d3d,
                         dilation: IntOrTuple2d3d,
                         stride: IntOrTuple2d3d) -> None:
    """
    Public validation for convolution hyperparameters.
    Intended for config validators and other early checks.
    """
    _validate_conv_inputs(kernel=kernel, dilation=dilation, stride=stride)

def _validate_param(value: IntOrTuple2d3d,
                    name: EncoderParamSpec) -> IntOrTuple2d3d:
    """
    Validate a convolution parameter that may be an int or a 2D/3D tuple.

    :param value: Parameter value to validate
    :param name: Parameter name to be used for error messages
    :return: Validated input value
    """
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{name.value} must be positive, got {value}")

    elif isinstance(value, tuple):
        if len(value) not in (2, 3):
            raise ValueError(f"Input {name.value} must have length 2 or 3, got {len(value)}")
        for i, v in enumerate(value):
            if not isinstance(v, int):
                raise ValueError(f"Input must contain only integers: {name.value}[{i}] must be int, got {type(v).__name__}")
            if v <= 0:
                raise ValueError(f"Input must contain only positive values: {name.value}[{i}] must be positive, got {v}")
    return value

def _normalize_stride(stride: IntOrTuple2d3d, num_dims: int) -> TupleOfInt2d3d:
    """
    Normalize the number of dimensions in input stride

    :param stride: Stride of the convolution operation, can be an int or a 2d or 3d tuple of int
    :param num_dims: Number of dimensions to convert to when all inputs are scalars, can be 2 or 3
    :return: Resulting stride as a tuple that has num_dims dimensions
    """
    if isinstance(stride, int):
        return expand_to_tuple(stride, num_dims)
    else:
        if len(stride) != num_dims:
            raise ValueError("Stride's tuple must have size = num_dims")
        return stride

def _normalize_kernel_dilation(kernel: IntOrTuple2d3d,
                               dilation: IntOrTuple2d3d,
                               num_dims: Optional[int] = None) -> Tuple[TupleOfInt2d3d, TupleOfInt2d3d]:
    """
    Normalize the number of dimensions in input kernel and dilation.

    - If all inputs are scalars: use num_dims to determine the number of dimensions, can be 2 or 3
    - If only ONE input is tuple: expand all scalars to match its size
    - If ALL inputs are tuples: they must all have the same size

    :param kernel: Kernel size of the convolution operation, can be an int or a 2d or 3d tuple of int
    :param dilation: Dilation factor of the convolution operation, can be an int or a 2d or 3d tuple of int
    :param num_dims: Number of dimensions to convert to when all inputs are scalars, can be 2 or 3

    :return: Resulting kernel and dilation as a tuple
    """
    if isinstance(kernel, int) and isinstance(dilation, int):
        if num_dims is None:
            raise ValueError("Both inputs are scalars, number of dimensions argument num_dims must be specified")
        kernel_value = expand_to_tuple(kernel, num_dims)
        dilation_value = expand_to_tuple(dilation, num_dims)

    elif isinstance(kernel, int) and isinstance(dilation, tuple):
        kernel_value = expand_to_tuple(kernel, len(dilation))
        dilation_value = dilation

    elif isinstance(dilation, int) and isinstance(kernel, tuple):
        dilation_value = expand_to_tuple(dilation, len(kernel))
        kernel_value = kernel

    elif isinstance(dilation, tuple) and isinstance(kernel, tuple):
        kernel_value = kernel
        dilation_value = dilation
        if len(kernel_value) != len(dilation_value):
            raise ValueError("All inputs must have the same length")
    else:
        raise RuntimeError("Input kernel and dilation must be int or tuples of int")

    return kernel_value, dilation_value

def _ensure_2d_params(**params: Any) -> None:
    """
    Raise ValueError if any convolution param is not 2D.
    Usage: _ensure_2d_params(kernel=self.kernel, stride=self.stride, dilation=self.dilation)
    """
    for name, value in params.items():
        if isinstance(value, tuple) and len(value) != 2:
            raise ValueError(f"{name} must be 2D")

def get_padding(kernel: TupleOfInt2d3d, dilation: TupleOfInt2d3d) -> TupleOfInt2d3d:
    """
    Calculate padding for encoder's convolutional layer as padding = (kernel - 1) * dilation // 2.
    For stride = 1, this will preserve the dimensions of the output. For stride = 2, it will downsample by a factor of 2.

    The same formula for padding is used in decoder's nn.ConvTranspose2d as per documentation,
    where the values of kernel and dilation are those used in the decoder.

    Note, a formula that is derived from the mathematical relationship between input and output pixel counts,
    produces incorrect output dimensions due to floor division.

    :return: Padding values
    """
    padding = tuple([(k - 1) * d // 2 for k, d in zip(kernel, dilation)])
    return cast(TupleOfInt2d3d, padding)

def get_encoder_params_2d(kernel: IntOrTuple2d, stride: IntOrTuple2d, dilation: IntOrTuple2d) -> EncoderParams2D:
    """
    Get validated and normalized convolution parameters for nn.Conv2d.

    :param kernel: Kernel size of the convolution operation, can be an int or a 2D tuple of int
    :param stride: Stride of the convolution operation, can be an int or a 2D tuple of int
    :param dilation: Dilation factor of the convolution operation, can be an int or a 2D tuple of int
    :return: An `EncoderParams2D` instance with validated kernel, stride, dilation, and computed padding.
             Safe to pass directly to 2D convolution layers.
    :raises ValueError: If any input is not 2D or contains invalid values
    """
    _ensure_2d_params(kernel=kernel, stride=stride, dilation=dilation)
    return EncoderParams.from_inputs(kernel=kernel, stride=stride, dilation=dilation, num_dims=2).as_2d()
