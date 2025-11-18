from __future__ import annotations
from enum import Enum, unique
from typing import List, Tuple, Union, Iterable, Any
from dataclasses import dataclass
from typing import Tuple

# Design note: This module adopts a strict, enum-first API for padding modes.
# Public functions expect a `PaddingMode` enum member (e.g. `PaddingMode.SAME`) and
# will raise ValueError for strings or other types. If you need lenient parsing
# (e.g. from CLI/config), add a thin adapter that converts strings to enum
# members at the boundary.


@dataclass(frozen=True)
class PaddingModeSpec:
    """
    Metadata for a padding mode used by the sampling utilities.

    :param value_str: Canonical string identifier for the mode (e.g. "same",
               "downsampling_factor_2").
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
    def from_value(cls, value) -> PaddingMode:
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


def ensure_padding_mode(value: Any) -> PaddingMode:
    """
    Validate and return a `PaddingMode` enum member.

    Behavior:
        - If `value` is already a `PaddingMode` member it is returned unchanged.
        - Otherwise, a ValueError is raised. This function enforces a strict policy:
        only PaddingMode enum members are accepted. If you want to allow strings
        (e.g., from user input or config files), convert them to enum members before calling

    # Example: User or config provides a string
    padding_mode_str = "same"  # could come from a config file, CLI, etc.

    # Convert string to enum at the boundary (I/O/config layer)
    try:
        padding_mode = PaddingMode[padding_mode_str.upper()]
    except KeyError:
        raise ValueError(f"Invalid padding mode: {padding_mode_str}")

    # Now use the strict API internally
    padding = get_padding(
        kernel=(3, 3),
        dilation=(1, 1),
        padding=padding_mode
    )

    Raises:
        ValueError: if `value` is not a `PaddingMode` member.
    """
    return PaddingMode.from_value(value)


def handle_mixed_inputs(kernel: Union[Iterable[int], int],
                        dilation: Union[Iterable[int], int],
                        num_dims: int = 2) -> Tuple[List[int], List[int]]:
    """
    Handle mixed inputs of different dimensions with strict validation:

    - If all inputs are scalars: use num_dims to determine the number of dimensions
    - If only ONE input is iterable: expand all scalars to match that iterable's size
    - If ALL inputs are iterables: they must all have the same size

    :param kernel: Kernel size of the convolution operation, can be an integer, a tuple or a list
    :param dilation: Dilation factor of the convolution operation, can be an integer, a tuple or a list
    :param num_dims: Number of dimensions to convert to when all inputs are scalars
    :raises ValueError: If inputs are invalid combinations of iterables and scalars, or contain zero/negative/empty values

    :return: Tuple of two lists containing the expanded kernel and dilation values
    """
    # Helper function to validate values
    def validate_value(value, name):
        if isinstance(value, Iterable) and not isinstance(value, str):
            if len(value) == 0:
                raise ValueError(f"Empty {name} is not allowed")
            for i, v in enumerate(value):
                if not isinstance(v, int):
                    raise TypeError(f"{name} must contain only integers, got {type(v).__name__} at index {i}")
                if v <= 0:
                    raise ValueError(f"{name} must contain only positive values, got {v} at index {i}")
        else:
            if not isinstance(value, int):
                raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

    validate_value(kernel, "kernel")
    validate_value(dilation, "dilation")
    kernel_is_iterable = isinstance(kernel, Iterable) and not isinstance(kernel, str)
    dilation_is_iterable = isinstance(dilation, Iterable) and not isinstance(dilation, str)

    # Case 1: All scalars - use num_dims
    if not kernel_is_iterable and not dilation_is_iterable:
        target_size = num_dims
        return ([kernel] * target_size, [dilation] * target_size)

    # Case 2: Only kernel is iterable - expand dilation to match kernel size
    elif kernel_is_iterable and not dilation_is_iterable:
        kernel_list = list(kernel)
        target_size = len(kernel_list)
        return (kernel_list, [dilation] * target_size)

    # Case 3: Only dilation is iterable - expand kernel to match dilation size
    elif not kernel_is_iterable and dilation_is_iterable:
        dilation_list = list(dilation)
        target_size = len(dilation_list)
        return ([kernel] * target_size, dilation_list)

    # Case 4: Both are iterables - they must have the same size
    else:
        kernel_list = list(kernel)
        dilation_list = list(dilation)

        if len(kernel_list) != len(dilation_list):
            raise ValueError(f"All iterable inputs must have the same size. "
                           f"Got kernel with size {len(kernel_list)} and dilation with size {len(dilation_list)}")
        return (kernel_list, dilation_list)


def get_padding_stride_2(kernel: Union[Iterable[int], int],
                         dilation: Union[Iterable[int], int]) -> Tuple[int, ...]:
    """
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

    :return: Padding value as a tuple of integers
    """
    kernel, dilation = handle_mixed_inputs(kernel, dilation, num_dims=2)

    for k in kernel:
            if k % 2 != 0:
                raise ValueError(f"Kernel size {k} is odd; for stride=2, even kernel sizes are required to avoid uneven overlap and artifacts.")


    padding = [((k - 1) * d) // 2 for k, d in zip(kernel, dilation)]
    return tuple(p for p in padding)


def get_padding(kernel: Union[Iterable[int], int],
                           dilation: Union[Iterable[int], int],
                           padding_mode: PaddingMode,
                           num_dims: int = 2) -> Union[Tuple[int, ...], str]:
    """
    Calculate padding value for convolution layers.

    Note: This function requires `padding_mode` to be a `PaddingMode` enum member. It
    will not accept free-form strings. Use `PaddingMode.SAME`,  etc.

    :param kernel: Kernel size of the convolution operation (int or iterable of int)
    :param dilation: Dilation factor of the convolution operation (int or iterable of int)
    :param padding_mode: A `PaddingMode` enum member (e.g. `PaddingMode.SAME`)
    :param num_dims: Number of dimensions to convert to when all inputs are scalars

    :return: Padding value as integers tuple or string
    :raises ValueError: If invalid padding mode is provided
    """
    padding_mode = ensure_padding_mode(padding_mode)
    kernel, dilation = handle_mixed_inputs(kernel, dilation, num_dims=num_dims)

    if padding_mode == PaddingMode.SAME:
        return 'same'

    if padding_mode == PaddingMode.DOWNSAMPLING_FACTOR_2:
        return get_padding_stride_2(kernel=kernel, dilation=dilation)