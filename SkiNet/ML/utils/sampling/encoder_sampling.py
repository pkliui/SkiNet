from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import NamedTuple, Optional, Union, cast

from SkiNet.ML.utils.typing_utils import IntOrTuple2d3d, TupleOfInt2d3d, expand_to_tuple

# Design note: This module adopts a strict, enum-first API for padding modes.
# Public functions expect a `PaddingMode` enum member (e.g. `PaddingMode.SAME`) and
# will raise ValueError for strings or other types. If you need lenient parsing
# (e.g. from CLI/config), add a thin adapter that converts strings to enum
# members at the boundary.


@dataclass(frozen=True)
class PaddingModeSpec:
    """
    Metadata for a padding mode used by the sampling utilities.

    :param value_str: Canonical string identifier for the mode (e.g. "same", "downsampling_factor_2").
    :param factor: Integer downsampling factor implied by the mode (1 = no downsample,
        2 = downsample by 2, ...). Used to compute upsampling kernel sizes.
    :param stride: Default stride to use for convolutions in this mode.

    Note:
        The `value_str` field is metadata only. This module's public API requires
        that callers pass a `PaddingMode` enum member rather than the raw
        `value` string; `from_value` will not accept the `value` string.
    """
    value_str: str
    factor: int
    stride: int


@unique
class PaddingMode(Enum):
    """
    Keeps possible padding modes to be used in e.g. BaseConvLayer2D's torch tensors

    Members hold a small PaddingModeSpec payload with a human-readable value string, the downsampling
    factor and a default stride.
    """
    SAME = PaddingModeSpec(value_str="same", factor=1, stride=1)
    DOWNSAMPLING_FACTOR_2 = PaddingModeSpec(value_str="downsampling_factor_2", factor=2, stride=2)

    @property
    def downsampling_factor(self) -> int:
        return int(self.value.factor)

    @property
    def stride(self) -> int:
        return self.value.stride

    @classmethod
    def from_value(cls, value: PaddingMode) -> PaddingMode:
        """
        Verify that input is a PaddingMode instance and return it.

        This strict helper accepts only a `PaddingMode` enum member and will
        raise ValueError for any other input (strings, ints, None, etc.).

        Examples:
            PaddingMode.from_value(PaddingMode.SAME)  # returns PaddingMode.SAME
            PaddingMode.from_value("SAME")          # raises ValueError

        Raises:
            ValueError: if `value` is not a PaddingMode member.
        """
        if isinstance(value, cls):
            return value
        raise ValueError(f"{value!r} is not a valid {cls.__name__}.")


def _validate_param(value: IntOrTuple2d3d,
                    name: str) -> IntOrTuple2d3d:
    """
    Validate a convolution parameter that may be an int or a 2D/3D tuple.

    This helper enforces the following rules:
    - Scalars must be positive integers
    - Tuples must have length 2 or 3
    - All tuple elements must be positive integers

    The validated value is returned unchanged, but invalid inputs raise
    descriptive exceptions.

    :param value: Parameter value to validate (int, 2-tuple, or 3-tuple)
    :param name: Parameter name used for error messages
    :raises ValueError: If values are non-positive or tuple length is invalid
    :raises TypeError: If values are not integers
    :return: The validated input value
    """
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        return value

    elif isinstance(value, tuple):
        if len(value) not in (2, 3):
            raise ValueError(f"{name} must have length 2 or 3, got {len(value)}")

        for i, v in enumerate(value):
            if not isinstance(v, int):
                raise TypeError(f"Input must contain only integers: {name}[{i}] must be int, got {type(v).__name__}")
            if v <= 0:
                raise ValueError(f"Input must contain only positive values: {name}[{i}] must be positive, got {v}")

        return value

    else:
        raise TypeError(f"{name} must contain only integers or tuples of integers")


class ConvParams(NamedTuple):
    """
    Validated convolution parameters.

    This value object represents convolution geometry after all inputs
    (scalars or tuples) have been normalized and validated.

    Invariants:
        - `kernel` and `dilation` are tuples of equal length
        - Length is either 2 (2D convolution) or 3 (3D convolution)
        - All values are positive integers

    Typical usage:
        params = ConvParams.from_inputs(kernel=3, dilation=1, num_dims=2)
        padding = params.padding(PaddingMode.DOWNSAMPLING_FACTOR_2)

    Design notes:
        - Instances of this class are assumed to be valid by construction.
        - All validation and shape normalization happens *before* creation,
          via `from_inputs()` (which delegates to `_normalize_conv_inputs`).
        - Methods on this class may rely on the invariants above and therefore
          do not repeat validation checks.

    Fields:
        kernel:
            Kernel size per spatial dimension (2D or 3D).
        dilation:
            Dilation factor per spatial dimension (2D or 3D).
    """
    kernel: TupleOfInt2d3d
    dilation: TupleOfInt2d3d

    @classmethod
    def from_inputs(cls,
                    kernel: IntOrTuple2d3d,
                    dilation: IntOrTuple2d3d,
                    num_dims: Optional[int] = None) -> "ConvParams":
        """
        Normalize and validate convolution parameters.

        This is a thin wrapper around `_normalize_conv_inputs` that makes
        ConvParams a self-contained value object.
        """
        return _normalize_conv_inputs(kernel, dilation, num_dims)

    def padding(self, mode: PaddingMode) -> Union[TupleOfInt2d3d, str]:
        """
        Compute the padding implied by these convolution parameters
        and the given padding mode.
        """
        mode = PaddingMode.from_value(mode)

        if mode == PaddingMode.SAME:
            return "same"

        elif mode == PaddingMode.DOWNSAMPLING_FACTOR_2:
            return _compute_stride2_padding(
                kernel=self.kernel,
                dilation=self.dilation,
            )
        else:
            raise ValueError(f"Unsupported padding mode: {mode}")


def _normalize_conv_inputs(kernel: IntOrTuple2d3d,
                           dilation: IntOrTuple2d3d,
                           num_dims: Optional[int] = None) -> ConvParams:
    """
    Handle mixed inputs of different dimensions with strict validation:

    - If all inputs are scalars: use num_dims to determine the number of dimensions, can be 2 or 3
    - If only ONE input is tuple: expand all scalars to match that iterable's size
    - If ALL inputs are tuples: they must all have the same size

    :param kernel: Kernel size of the convolution operation, can be an int or a 2d or 3d tuple of int
    :param dilation: Dilation factor of the convolution operation, can be an int or a 2d or 3d tuple of int
    :param num_dims: Number of dimensions to convert to when all inputs are scalars, can be 2 or 3
    :raises ValueError: If inputs are invalid combinations of iterables and scalars, or contain zero/negative/empty values

    :return: Named tuple of expanded kernel and dilation values
    """

    if num_dims is not None and num_dims not in {2, 3}:
        raise ValueError("Invalid number of dimensions, only 2d and 3d datasets are supported")

    kernel = _validate_param(kernel, "kernel")
    dilation = _validate_param(dilation, "dilation")

    if isinstance(kernel, int) and isinstance(dilation, int):
        if num_dims is None:
            raise ValueError("Both inputs are scalars, number of dimensions argument num_dims must be specified")
        kernel_value = expand_to_tuple(kernel, num_dims)
        dilation_value = expand_to_tuple(dilation, num_dims)

    elif isinstance(kernel, int) and not isinstance(dilation, int):
        kernel_value = expand_to_tuple(kernel, len(dilation))
        dilation_value = dilation

    elif isinstance(dilation, int) and not isinstance(kernel, int):
        dilation_value = expand_to_tuple(dilation, len(kernel))
        kernel_value = kernel

    elif isinstance(dilation, tuple) and isinstance(kernel, tuple):
        kernel_value = kernel
        dilation_value = dilation

        if len(kernel_value) != len(dilation_value):
            raise ValueError("All inputs must have the same length")
    else:
        raise RuntimeError("Unreachable state in _normalize_conv_inputs. Input kernel and dilation must be int or tuples of int")

    return ConvParams(kernel=kernel_value, dilation=dilation_value)


def _compute_stride2_padding(kernel: TupleOfInt2d3d,
                             dilation: TupleOfInt2d3d) -> TupleOfInt2d3d:
    """
    Expected to be called via e.g. get_padding.

    Calculate padding for stride=2 convolutions. Output will be downsampled by factor 2.
    Sampling mode is 'DOWNSAMPLING_FACTOR_2'.

    In general, stride, dilation and kernel sizes are constrained to have the following values,
    ensuring that the resulting padding, p=((k - 1) * d - 1) // 2,
    is an integer for even and odd inputs: s=2, k=2n, d=2n+1; n=1,2,3,...

    Here, the formula for padding, p=(k - 1) * d // 2,
    achieves exact downsampling by factor of 2 in torch.nn.conv2d for even inputs.
    It differs from the exact formula, p=((k - 1) * d - 1) // 2, due to floor division.

    :param kernel: Kernel size of the convolution operation
    :param dilation: Dilation factor of the convolution operation

    :return: Padding values
    """

    for i, k in enumerate(kernel):
        if k % 2 != 0:
            raise ValueError(f"Kernel size {k} is odd; for stride=2, even kernel sizes are required to avoid uneven overlap and artifacts.")

    padding = tuple((k - 1) * d // 2 for k, d in zip(kernel, dilation))
    return cast(TupleOfInt2d3d, padding)


def get_padding(kernel: IntOrTuple2d3d,
                dilation: IntOrTuple2d3d,
                padding_mode: PaddingMode,
                num_dims: int = 2) -> Union[TupleOfInt2d3d, str]:
    """
    Calculate padding value for convolution layers.

    :param kernel: Kernel size of the convolution operation (int or iterable of int)
    :param dilation: Dilation factor of the convolution operation (int or iterable of int)
    :param padding_mode: A `PaddingMode` enum member (e.g. `PaddingMode.SAME`)
    :param num_dims: Number of dimensions to convert to when all inputs are scalars

    :return: Padding value as integers tuple or string
    """
    params = ConvParams.from_inputs(kernel, dilation, num_dims)
    return params.padding(padding_mode)
