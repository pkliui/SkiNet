from SkiNet.ML.utils.sampling.encoder_sampling import ConvParamSpec, _validate_param
from SkiNet.ML.utils.typing_utils import IntOrTuple2d3d


def get_padding_for_transpose_conv(kernel: IntOrTuple2d3d) -> int:
    """
    Return padding required for torch.nn.ConvTranspose2d

    Input validated through encoder _validate_param
    :param kernel: Kernel size of the convolution operation in the encoder

    :return output_padding value that is either 0 or 1
    """
    _validate_param(kernel, ConvParamSpec.KERNEL)
    return (kernel - 1) // 2

def get_output_padding(kernel: IntOrTuple2d3d, stride: IntOrTuple2d3d) -> int:
    """
    Return output_padding required for torch.nn.ConvTranspose2d

    Input validated through encoder _validate_param
    :param kernel: Kernel size of the convolution operation, equal to that used in the encoder
    :param stride: Stride of the convolution operation, equal to that used in the encoder

    :return output_padding value that is either 0 or 1
    """
    _validate_param(kernel, ConvParamSpec.KERNEL)
    _validate_param(stride, ConvParamSpec.STRIDE)
    if stride % 2 == 0:
        return 0 if kernel % 2 == 0 else 1
    else:
        return 0
