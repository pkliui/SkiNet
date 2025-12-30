from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import Optional, Tuple, cast

from SkiNet.ML.utils.typing_utils import IntOrTuple2d3d, TupleOfInt2d3d, expand_to_tuple


@unique
class ConvParamSpec(Enum):
    KERNEL = "Size of the convolving kernel"
    DILATION = "Spacing between convolving kernels"
    STRIDE = "Stride of the convolution"

@dataclass(frozen=True)
class ConvParams:
    """
    Validated and normalized convolution parameters.

    The dimensions of kernels and dilations must be the same in order to compute padding.
    Strides do not need to match them - torch will use the same stride value
    for the height and width automatically, unless strides are tuples.

    :param kernel: Size of the convolving kernel
    :param dilation: Spacing between convolving kernels
    :param stride: Stride of the convolution
    """
    kernel: TupleOfInt2d3d
    dilation: TupleOfInt2d3d
    stride: IntOrTuple2d3d

    @classmethod
    def from_inputs(cls,
                    kernel: IntOrTuple2d3d,
                    dilation: IntOrTuple2d3d,
                    stride: IntOrTuple2d3d,
                    num_dims: Optional[int] = None) -> "ConvParams":
        """
        Validate and normalize convolution parameters.
        """
        _validate_conv_inputs(kernel, dilation, stride)
        kernel, dilation = _normalize_kernel_dilation(kernel, dilation, num_dims)
        params = cls(kernel=kernel, dilation=dilation, stride=stride)
        params._ensure_integer_padding()
        return params

    def _ensure_integer_padding(self) -> None:
        """
        Ensure integer padding so that the shape of the output is preserved for stride=1 and is halved exactly for stride=2.
        Padding is computed as (kernel - 1)*dilation // 2, the output shape is given by
        https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        `out = [((in + 2*padding - dilation*(kernel-1) - 1)/stride)+ 1]`

        Even kernels and odd dilations will raise the error.
        """
        for i, (k, d) in enumerate(zip(self.kernel, self.dilation)):
            if (k - 1) * d % 2 != 0:
                raise ValueError(f"Incompatible kernel/dilation at dim {i}: "
                                 f"(k={k}, d={d}) leads to fractional padding")


def _validate_conv_inputs(kernel: IntOrTuple2d3d,
                          dilation: IntOrTuple2d3d,
                          stride: IntOrTuple2d3d) -> None:

    kernel = _validate_param(kernel, ConvParamSpec.KERNEL)
    dilation = _validate_param(dilation, ConvParamSpec.DILATION)
    stride = _validate_param(stride, ConvParamSpec.STRIDE)


def _validate_param(value: IntOrTuple2d3d,
                    name: ConvParamSpec) -> IntOrTuple2d3d:
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
    if num_dims is not None and num_dims not in {2, 3}:
        raise ValueError("Invalid number of dimensions, only 2d and 3d datasets are supported")

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
        raise RuntimeError("Unreachable state in _normalize_conv_inputs. Input kernel and dilation must be int or tuples of int")

    return kernel_value, dilation_value


def get_padding(convparams: ConvParams) -> TupleOfInt2d3d:
    """
    Calculate padding as per empirical formula (kernel - 1) * dilation // 2.

    The exact formula, derived from the mathematical relationship between input and output pixel counts,
    produces incorrect output dimensions when reused for both stride = 1 and stride = 2 convolutions.

    :param convparams: An instance of validated and normalized convolution parameters,
         obtained by calling convparams=ConvParams.from_inputs(kernel, dilation, stride)
    :param param.kernel: Kernel size of the convolution operation
    :param param.dilation: Dilation factor of the convolution operation

    :return: Padding values as a tuple
    """
    padding = tuple([(k - 1)*d // 2 for k, d in zip(convparams.kernel, convparams.dilation)])
    assert len(padding) in (2, 3)
    return cast(TupleOfInt2d3d, padding)
