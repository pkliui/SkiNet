import pytest

from SkiNet.ML.utils.sampling.decoder_sampling import get_decoder_params_2d
from SkiNet.ML.utils.sampling.encoder_sampling import get_encoder_params_2d
from SkiNet.ML.utils.typing_utils import IntOrTuple2d, TupleOfInt2d


@pytest.mark.parametrize(
    "stride, kernel, dilation, expected_decoder_stride, expected_decoder_kernel, expected_decoder_dilation, expected_decoder_padding",
    [
        (2, (3, 3), (1, 1), (2, 2), (6, 6), (1, 1), (2, 2)),  # 1d stride, d=1
        ((2, 2), (5, 5), (1, 1), (2, 2), (10, 10), (1, 1), (4, 4)),  # 2d stride d=1
        ((2, 2), 3, (2, 2), (2, 2), (6, 6), (2, 2), (5, 5)),  # 1d kernel, d=2
        ((2, 2), (5, 5), (2, 2), (2, 2), (10, 10), (2, 2), (9, 9)),  # 2d kernel d=2

    ],
)
def test_decoder_params_2d_returns(stride: IntOrTuple2d, kernel: IntOrTuple2d, dilation: IntOrTuple2d,
                                   expected_decoder_stride: TupleOfInt2d, expected_decoder_kernel: TupleOfInt2d,
                                   expected_decoder_dilation: TupleOfInt2d, expected_decoder_padding: TupleOfInt2d) -> None:
    """
    Test the value of decoder's stride, kernel, dilation, padding obtained from provided encoders's input parameters
    """
    conv_params = get_encoder_params_2d(kernel=kernel, stride=stride, dilation=dilation)
    decoder = get_decoder_params_2d(conv_params)

    assert decoder.stride == expected_decoder_stride
    assert decoder.kernel == expected_decoder_kernel
    assert decoder.dilation == expected_decoder_dilation
    assert decoder.padding == expected_decoder_padding


@pytest.mark.parametrize(
    "kernel, stride, dilation, expected_output_padding",
    [
        ((3, 3), 2, 1, (0, 0)),  # k_encoder=3, k_decoder=k_encoder*s=3*2=6, d(k-1)=1*5=5
        ((3, 3), 2, 2, (1, 1)),  # k_encoder = 3, k_decoder=k_encoder*s=3*2=6, d(k-1)=2*6=12
        ((3, 3), 1, 1, (1, 1)),  # k_encoder = 3, k_decoder=k_encoder*s=3*1=3, d(k-1)=1*2=2
        ((3, 3), 1, 2, (1, 1)),  # k_encoder = 3, k_decoder=k_encoder*s=3*2=6, d(k-1)=2*5=10
    ],
)
def test_decoder_params_2d_output_padding(kernel: IntOrTuple2d, stride: IntOrTuple2d, dilation: IntOrTuple2d, expected_output_padding: TupleOfInt2d) -> None:
    """
    Test the value of decoder.padding  obtained from provided encoders's input parameters
    """
    conv_params = get_encoder_params_2d(kernel=kernel, stride=stride, dilation=dilation)
    decoder = get_decoder_params_2d(conv_params)

    assert decoder.output_padding == expected_output_padding
