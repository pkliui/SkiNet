from enum import Enum, unique
from typing import Tuple, Union, Iterable
from typing_extensions import Sized


@unique
class PaddingMode(Enum):
    """
    Keeps possible padding modes to be used in e.g. BaseConvLayer2D's torch tensors

    :param VALID: No padding applied to input. The resulting output size is determined by the kernel size, dilation and stride.
    :param SAME: Apply padding such that input and output have the same shape.
    :param DOWNSAMPLING_FACTOR_2: Apply padding such that the input is downsampled by a factor of 2.
    """
    VALID = "valid"
    SAME = "same"
    DOWNSAMPLING_FACTOR_2 = "downsampling_factor_2"
    

def handle_mixed_inputs(kernel: Union[Iterable[int], int], 
                        dilation: Union[Iterable[int], int], 
                        num_dims: int = 2) -> Tuple[Union[Iterable[int], int], Union[Iterable[int], int], Union[Iterable[int], int]]:
    """
    Handle mixed inputs of different dimensions with strict validation:
    
    - If all inputs are scalars: use num_dims to determine the number of dimensions
    - If only ONE input is iterable: expand all scalars to match that iterable's size
    - If ALL inputs are iterables: they must all have the same size
    - Raises ValueError for invalid combinations, zero values, empty lists, or negative values

    :param kernel: Kernel size of the convolution operation, can be an integer, a tuple or a list
    :param dilation: Dilation factor of the convolution operation, can be an integer, a tuple or a list
    :param num_dims: Number of dimensions to convert to when all inputs are scalars
    :raises ValueError: If inputs are invalid combinations of iterables and scalars, or contain zero/negative/empty values
    """
    
    # Helper function to validate values
    def validate_value(value, name):
        if isinstance(value, Iterable):
            if len(value) == 0:
                raise ValueError(f"Empty {name} is not allowed")
            for i, v in enumerate(value):
                if v <= 0:
                    raise ValueError(f"{name} must contain only positive values, got {v} at index {i}")
        else:
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


def get_same_padding_stride_1(kernel: Union[Iterable[int], int], dilation: Union[Iterable[int], int]) -> Union[Iterable[int], int]:
    """
    Calculate padding for stride=1 convolutions so that output has same size as input, using formula
    padding = (kernel-1)*dilation/2, 
    
    where stride, kernel and dilation sizes are constrained to the following values, 
    ensuring that padding is an integer:
    s=1, k=2n+1, d=1; n=1,2,3,...
    s=1, k=n, d=2; n=1,2,3,...

    :param kernel: Kernel size of the convolution operation
    :param dilation: Dilation factor of the convolution operation

    :return: Padding value as integer

    """
    
    # Validate inputs using handle_mixed_inputs
    kernel, dilation = handle_mixed_inputs(kernel, dilation, num_dims=2)

    padding = [(k - 1) * d / 2 for k, d in zip(kernel, dilation)]

    if not all(p.is_integer() for p in padding):
        raise ValueError(f"Invalid input. Given kernel size={kernel}, dilation={dilation}, not all items in padding {padding} are integers.")
    return tuple(int(p) for p in padding)


def get_padding_stride_2(kernel: Union[Iterable[int], int], dilation: Union[Iterable[int], int]) -> Union[Iterable[int], int]:
    """
    Calculate padding for stride=2 convolutions. Output will be downsampled by factor 2.
    Sampling mode is 'DOWNSAMPLING_FACTOR_2'.

    In general, stride, dilation and kernel sizes are constrained to have the following values, ensuring that the resulting padding is an integer:
    s=2, k=2n, d=2n+1; n=1,2,3,...

    :param kernel: Kernel size of the convolution operation
    :param dilation: Dilation factor of the convolution operation

    :return: Padding value as float
    """
    
    # Validate inputs using handle_mixed_inputs
    kernel, dilation = handle_mixed_inputs(kernel, dilation, num_dims=2)

    padding = [((k - 1) * d - 1) / 2 for k, d in zip(kernel, dilation)]
    
    if not all(p.is_integer() for p in padding):
        raise ValueError(f"Invalid input. Given kernel size={kernel}, dilation={dilation}, not all items in padding {padding} are integers.")
    return tuple(int(p) for p in padding)

def get_padding_value(stride: int,
                      kernel: int,
                      dilation: int,
                      padding_mode: PaddingMode, 
                      num_dims: int = 2) -> Tuple[int, ...]:
    """
    Calculate padding value for convolution layers based on kernel size, stride, dilation,
    and sampling mode.

    In general, stride, dilation and kernel sizes are constrained to have the following values, ensuring that the resulting padding is an integer:
    - SAME: 
        s=1, k=2n+1, d=1; n=1,2,3,...
        s=1, k=any, d=2
    - DOWNSAMPLING_FACTOR_2:
        s=2, k=2n, d=2n+1; n=1,2,3,...

    - s=2, d=2n are not supported as they result in odd padding values for any kernel size.

    Args:
        stride: Stride of the convolution operation
        kernel: Kernel size of the convolution operation
        dilation: Dilation factor of the convolution operation
        padding_mode: Sampling strategy (VALID, SAME, or DOWNSAMPLING_FACTOR_2)
        num_dims: Number of dimensions to convert to when all inputs are scalars
    
    Returns:
        Padding value as integer
    
    Raises:
        ValueError: If invalid parameter combinations are provided or padding is not integer
    """

    if padding_mode not in [mode for mode in PaddingMode]:
        raise ValueError(f"Invalid sampling mode: {padding_mode}")
    
    # Validate stride value
    if stride not in [1, 2]:
        raise ValueError(f"Stride must be 1 or 2, got {stride}")

    # ensure the same size for kernel and dilation
    kernel, dilation = handle_mixed_inputs(kernel, dilation, num_dims=2)


    if padding_mode == PaddingMode.VALID:
        return tuple([0] * num_dims)
    
    # keep output size equal to input size
    elif padding_mode == PaddingMode.SAME:
        if stride == 1:
            return get_same_padding_stride_1(kernel=kernel, dilation=dilation)
        else:
            raise ValueError(f"Unsupported stride {stride} for SAME sampling. Only stride=1 is supported.")
    
    # downsample the output by factor of 2
    elif padding_mode == PaddingMode.DOWNSAMPLING_FACTOR_2:
        if stride == 2:
            return get_padding_stride_2(kernel=kernel, dilation=dilation)
        else:
            raise ValueError(f"Unsupported stride {stride} for DOWNSAMPLING_FACTOR_2 sampling. Only stride=2 is supported.")
    


