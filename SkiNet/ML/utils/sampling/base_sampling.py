from typing import Protocol

from SkiNet.ML.utils.typing_utils import TupleOfInt2d, TupleOfInt2d3d


class BaseEncoderParams(Protocol):
    """
    Protocol for convolution parameters in encoder
    """
    kernel: TupleOfInt2d3d
    stride: TupleOfInt2d3d
    dilation: TupleOfInt2d3d
    padding: TupleOfInt2d3d

class EncoderParams2D(BaseEncoderParams):
    """
    Protocol for convolution parameters for nn.Conv2d layer
    """
    kernel: TupleOfInt2d
    stride: TupleOfInt2d
    dilation: TupleOfInt2d
    padding: TupleOfInt2d


class BaseDecoderParams(Protocol):
    """
    Protocol for convolution parameters in decoder
    """
    kernel: TupleOfInt2d3d
    stride: TupleOfInt2d3d
    dilation: TupleOfInt2d3d
    padding: TupleOfInt2d3d
    output_padding: TupleOfInt2d3d

class DecoderParams2D(BaseDecoderParams):
    """
    Protocol for convolution parameters for nn.ConvTranspose2d
    """
    kernel: TupleOfInt2d
    stride: TupleOfInt2d
    dilation: TupleOfInt2d
    padding: TupleOfInt2d
    output_padding: TupleOfInt2d
