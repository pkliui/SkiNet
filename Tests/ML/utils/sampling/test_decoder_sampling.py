import pytest

from SkiNet.ML.utils.sampling.decoder_sampling import get_output_padding, get_padding_for_transpose_conv
from SkiNet.ML.utils.typing_utils import IntOrTuple2d3d


@pytest.mark.parametrize(
    "kernel, msg",
    [
        (-1, "must be positive"),
        ("1", "must be int or tuple"),
        (1.5, "must be int or tuple"),
        (0, "must be positive"),
        ((0, 3), "must be positive"),
        ((3, 0), "must be positive"),
        ((-1, 3), "must be positive"),
        ((3, -1), "must be positive"),
        ((3.5, 2), "must be int"),
        (("3", 2), "must be int"),
        ((1, 2, 3, 4), "must have length 2 or 3"),  # invalid length
    ],
)
def test_get_padding_for_transpose_conv_invalid(kernel: IntOrTuple2d3d, msg: str) -> None:
    with pytest.raises(ValueError, match=msg):
        get_padding_for_transpose_conv(kernel)


@pytest.mark.parametrize(
    "kernel,expected_padding",
    [
        (1, 0),
        (2, 0),
        (3, 1),
        (4, 1),
        (5, 2),
        (6, 2),
        (7, 3),
    ],
)
def test_get_padding_for_transpose_conv(kernel: IntOrTuple2d3d,
                                        expected_padding: IntOrTuple2d3d) -> None:
    padding = get_padding_for_transpose_conv(kernel)

    assert isinstance(padding, IntOrTuple2d3d)
    assert padding == expected_padding


@pytest.mark.parametrize(
    "kernel, stride, msg",
    [
        (1, -1, "must be positive"),
        (1, "2", "must be int or tuple"),
        (1, 1.5, "must be int or tuple"),
        (1, 0, "must be positive"),
        ((3, 3), -1, "must be positive"),
        ((3, -3), 1, "must be positive"),
        ((3, 3), -2, "must be positive"),
        ((-3, 3), 2, "must be positive"),
        ((3, 3), 2.2, "must be int"),
        ((3, 3), "1", "must be int"),
        ((1, 1), (1, 1, 1, 1), "must have length 2 or 3"),  # invalid length
    ],
)
def test_get_output_padding_invalid(kernel: IntOrTuple2d3d, stride: IntOrTuple2d3d, msg: str) -> None:
    with pytest.raises(ValueError, match=msg):
        _ = get_output_padding(kernel, stride)

@pytest.mark.parametrize(
    "kernel,stride,expected_output_padding",
    [
        # stride even
        (2, 2, 0),
        (3, 2, 1),
        (4, 2, 0),
        (5, 2, 1),

        # stride odd
        (2, 1, 0),
        (3, 1, 0),
        (4, 3, 0),
        (5, 5, 0),
    ],
)
def test_get_output_padding(kernel: IntOrTuple2d3d,
                            stride: IntOrTuple2d3d,
                            expected_output_padding: IntOrTuple2d3d) -> None:
    output_padding = get_output_padding(kernel, stride)

    assert isinstance(output_padding, IntOrTuple2d3d)
    assert output_padding == expected_output_padding
