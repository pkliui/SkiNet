from enum import Enum, unique
from typing import List, Optional, Tuple, Union, Iterable
from dataclasses import dataclass
from typing import Tuple


@unique
class PaddingMode(Enum):
    """
    Keeps possible padding modes to be used in e.g. BaseConvLayer2D's torch tensors

    :param VALID: No padding applied to input. The resulting output size is determined by the kernel size, dilation and stride.
    :param SAME: Apply padding such that input and output have the same shape. This is only supported for stride=1 convolutions.
    :param DOWNSAMPLING_FACTOR_2: Apply padding such that the input is downsampled by a factor of 2. This is only supported for stride=2 convolutions.
    """
    VALID = "valid"
    SAME = "same"
    DOWNSAMPLING_FACTOR_2 = "downsampling_factor_2"


@dataclass
class ConvParams:
    """
    Holds the parameters to be returned by get_padding_and_stride
    :param padding: Padding value as a tuple of integers
    :param stride: Stride value as an integer, specific to the padding mode passed to get_padding_and_stride
    """
    padding: Tuple[int, ...]
    stride: int
    

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
    
    # Validate inputs
    validate_value(kernel, "kernel")
    validate_value(dilation, "dilation")
    
    # Check which inputs are iterables
    kernel_is_iterable = isinstance(kernel, Iterable)
    dilation_is_iterable = isinstance(dilation, Iterable)
    
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

def get_same_padding_stride_1(kernel: Union[Iterable[int], int], dilation: Union[Iterable[int], int]) -> Tuple[int, ...]:
    """
    Calculate padding for stride=1 convolutions so that output has same size as input, using formula
    padding = (kernel-1)*dilation/2, 
    
    where stride, kernel and dilation sizes are constrained to the following values, 
    ensuring that padding is an integer:
    s=1, k=2n+1, d=1; n=1,2,3,...
    s=1, k=n, d=2; n=1,2,3,...

    :param kernel: Kernel size of the convolution operation
    :param dilation: Dilation factor of the convolution operation

    :return: Padding value as a tuple of integers
    """
    # Validate inputs using handle_mixed_inputs
    kernel, dilation = handle_mixed_inputs(kernel, dilation, num_dims=2)

    padding = [(k - 1) * d / 2 for k, d in zip(kernel, dilation)]

    if not all(p.is_integer() for p in padding):
        raise ValueError(f"Invalid input. Given kernel size={kernel}, dilation={dilation}, not all items in padding {padding} are integers.")
    
    return tuple(int(p) for p in padding)

def get_padding_stride_2(kernel: Union[Iterable[int], int], dilation: Union[Iterable[int], int]) -> Tuple[int, ...]:
    """
    Calculate padding for stride=2 convolutions. Output will be downsampled by factor 2.
    Sampling mode is 'DOWNSAMPLING_FACTOR_2'.

    In general, stride, dilation and kernel sizes are constrained to have the following values, 
    ensuring that the resulting padding is an integer:
    s=2, k=2n, d=2n+1; n=1,2,3,...

    :param kernel: Kernel size of the convolution operation
    :param dilation: Dilation factor of the convolution operation

    :return: Padding value as a tuple of integers
    """
    # Validate inputs using handle_mixed_inputs
    kernel, dilation = handle_mixed_inputs(kernel, dilation, num_dims=2)

    padding = [((k - 1) * d - 1) / 2 for k, d in zip(kernel, dilation)]
    
    if not all(p.is_integer() for p in padding):
        raise ValueError(f"Invalid input. Given kernel size={kernel}, dilation={dilation}, not all items in padding {padding} are integers.")
    
    return tuple(int(p) for p in padding)

def get_padding_and_stride(
    kernel: Union[Iterable[int], int],
    dilation: Union[Iterable[int], int],
    padding_mode: PaddingMode, 
    num_dims: int = 2,
    stride: Optional[int] = None
) -> ConvParams:
    """
    Calculate padding value for convolution layers based on kernel size, stride, dilation,
    and sampling mode.

    In general, stride, dilation and kernel sizes are constrained to have the following values, ensuring that the resulting padding is an integer:
    - SAME (output has same size as input): 
        s=1, k=2n+1, d=1; n=1,2,3,...
        s=1, k=any, d=2
    - DOWNSAMPLING_FACTOR_2 (output is downsampled by factor 2):
        s=2, k=2n, d=2n+1; n=1,2,3,...

    - s=2, d=2n are not supported as they result in odd padding values for any kernel size.

    :param kernel: Kernel size of the convolution operation (int or iterable of int)
    :param dilation: Dilation factor of the convolution operation (int or iterable of int)
    :param padding_mode: Sampling strategy is VALID (no padding applied), SAME (output has same size as input), or DOWNSAMPLING_FACTOR_2 (output is downsampled by factor 2)
    :param num_dims: Number of dimensions to convert to when all inputs are scalars
    :param stride: Stride of the convolution operation for padding mode VALID. 
        If None, its value will be set based on padding_mode: 1 for SAME, 2 for DOWNSAMPLING_FACTOR_2.

    :return: Padding value as integer

    :raises ValueError: If invalid parameter combinations are provided or padding is not integer
    """
    # ensure the same size for kernel and dilation
    kernel, dilation = handle_mixed_inputs(kernel, dilation, num_dims=2)

    if padding_mode == PaddingMode.VALID:
        if stride is None:
            raise ValueError("Stride must be specified for VALID padding mode.")
        padding=tuple([0] * num_dims)
    
    # keep output size equal to input size
    elif padding_mode == PaddingMode.SAME:
        stride = 1
        padding=get_same_padding_stride_1(kernel=kernel, dilation=dilation)

    # downsample the output by factor of 2
    elif padding_mode == PaddingMode.DOWNSAMPLING_FACTOR_2:
        stride = 2
        padding=get_padding_stride_2(kernel=kernel, dilation=dilation)
    else:
        raise ValueError(f"Invalid sampling mode: {padding_mode}")
    
    return ConvParams(padding=padding, stride=stride)