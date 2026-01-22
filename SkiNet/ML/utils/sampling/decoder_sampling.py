from dataclasses import dataclass
from typing import cast

from SkiNet.ML.utils.sampling.base_sampling import BaseEncoderParams, DecoderParams2D, EncoderParams2D
from SkiNet.ML.utils.sampling.encoder_sampling import get_padding
from SkiNet.ML.utils.typing_utils import TupleOfInt2d3d


@dataclass(frozen=True)
class DecoderParams:
    """
    Decoder-side convolution parameters derived from an encoder's parameters.

    This class is mostly intended to be used via `get_decoder_params_2d` or similar factory functions
    to ensure proper validation and type safety.

    :param encoder: An instance of validated and normalized encoder convolution parameters
                    (typically `EncoderParams2D` or `EncoderParams`)
    :ivar kernel: Decoder's  kernel
    :ivar stride: Decoder's stride
    :ivar dilation: Decoder's dilation
    :ivar padding: Decoder's padding, computed using decoder's kernel and dilation
    :ivar output_padding: Extra padding used for `torch.nn.ConvTranspose2d` to achieve the desired output shape
    """
    encoder: BaseEncoderParams

    @property
    def stride(self) -> TupleOfInt2d3d:
        """
        Decoder's stride is that of the encoder
        """
        return self.encoder.stride

    @property
    def kernel(self) -> TupleOfInt2d3d:
        """
        Decoder's kernel is encoder.kernel * encoder.stride to avoid checkerboard artifacts
        https://distill.pub/2016/deconv-checkerboard/#citation
        """
        return cast(TupleOfInt2d3d,
                    tuple(k*s for k, s in zip(self.encoder.kernel, self.encoder.stride)))

    @property
    def dilation(self) -> TupleOfInt2d3d:
        """
        Decoder's dilation is that of the encoder
        """
        return self.encoder.dilation

    @property
    def padding(self) -> TupleOfInt2d3d:
        """
        Decoder's padding is computed using the same formula as for the encoder, but using decoder's kernel and dilation values
        """
        return get_padding(self.kernel, self.dilation)

    @property
    def output_padding(self) -> TupleOfInt2d3d:
        """
        Output padding used in torch.nn.ConvTranspose2d
        """
        return cast(TupleOfInt2d3d,
                    tuple(1 if (d * (k - 1)) % 2 == 0 else 0 for k, d in zip(self.kernel, self.dilation)))

    def as_2d(self) -> DecoderParams2D:
        """
        Return a 2D view of this DecoderParams.
        """
        return cast(DecoderParams2D, self)


def get_decoder_params_2d(conv_params: EncoderParams2D) -> DecoderParams2D:
    """
    Get convolution parameters for torch.nn.ConvTranspose2d from validated and normalized
    encoder's parameters.

    :param conv_params: An `EncoderParams2D` instance with validated kernel, stride, dilation, and computed padding.
    :return: A `DecoderParams2D` instance with validated kernel, stride, dilation, and computed padding and output padding.
             Safe to pass directly to transposed 2D convolution layers.
    """
    return DecoderParams(encoder=conv_params).as_2d()
