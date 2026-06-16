from typing import Any, Union

import pytest

from SkiNet.ML.utils.sampling.encoder_sampling import (EncoderParams, EncoderParamSpec, _normalize_kernel_dilation,
                                                       _validate_param, get_encoder_params_2d)
from SkiNet.ML.utils.typing_utils import IntOrTuple2d, IntOrTuple2d3d, TupleOfInt2d, TupleOfInt3d

# ---------------------------- Tests for _normalize_conv_inputs----------------------------

class TestHandleMixedInputs:
    """Test cases for _normalize_conv_inputs function"""

    def test_all_scalars_uses_num_dims(self) -> None:
        """Test that when all inputs are scalars, num_dims=3 is used"""
        kernel, dilation = _normalize_kernel_dilation(3, 1, 3)

        assert kernel == (3, 3, 3)
        assert dilation == (1, 1, 1)

    def test_all_scalars_default_num_dims(self) -> None:
        """Test that when all inputs are scalars, num_dims = 2 is used"""
        kernel, dilation = _normalize_kernel_dilation(5, 2, 2)

        assert kernel == (5, 5)
        assert dilation == (2, 2)

    def test_scalar_with_iterable(self) -> None:
        """Test scalar with iterable input"""
        kernel, dilation = _normalize_kernel_dilation(3, (1, 2, 3))

        assert kernel == (3, 3, 3)
        assert dilation == (1, 2, 3)

    def test_iterable_with_scalar(self) -> None:
        """Test iterable with scalar input"""
        kernel, dilation = _normalize_kernel_dilation((3, 5, 7), 2)

        assert kernel == (3, 5, 7)
        assert dilation == (2, 2, 2)

    def test_num_dims_ignored_when_iterables_present(self) -> None:
        """Test that num_dims is ignored when iterables are present"""
        kernel, dilation = _normalize_kernel_dilation((3, 5), 1, 3)

        # Should use max length (2) from the iterable, not num_dims (3)!
        assert kernel == (3, 5)
        assert dilation == (1, 1)
        assert len(kernel) == 2
        assert len(dilation) == 2

# --------------------------- Tests for _validate_param --------------------------

@pytest.mark.parametrize(
    "value,name",
    [
        (1, EncoderParamSpec.KERNEL),
        ((2, 2), EncoderParamSpec.KERNEL),
        ((3, 3), EncoderParamSpec.KERNEL),
        ((1, 2), EncoderParamSpec.KERNEL),
        ((1, 2, 2), EncoderParamSpec.KERNEL),
        ((1, 2, 3), EncoderParamSpec.KERNEL),
        (1, EncoderParamSpec.DILATION),
        ((1, 1), EncoderParamSpec.DILATION),
        ((2, 2), EncoderParamSpec.DILATION),
        ((1, 2, 2), EncoderParamSpec.DILATION),
        ((1, 2, 3), EncoderParamSpec.DILATION),
        (1, EncoderParamSpec.STRIDE),
        ((1, 1), EncoderParamSpec.STRIDE),
        ((2, 2), EncoderParamSpec.STRIDE),
        ((1, 2, 2), EncoderParamSpec.STRIDE)
    ]
)
def test_validate_param_valid(value: IntOrTuple2d3d, name: EncoderParamSpec) -> None:
    """
    Test that various valid inputs passed to _validate_param do not raise
    """
    _validate_param(value, name)

@pytest.mark.parametrize(
    "value,name,exc_msg",
    [
        (-1, EncoderParamSpec.KERNEL, "must be positive"),       # negative int
        (0, EncoderParamSpec.DILATION, "must be positive"),      # zero int
        ((1,), EncoderParamSpec.KERNEL, "must have length 2 or 3"),  # tuple too short
        ((1, 1, 1, 1), EncoderParamSpec.DILATION, "must have length 2 or 3"),  # tuple too long
        ((1, "a"), EncoderParamSpec.STRIDE, "must be int"),       # non-int element
        ((1, -1), EncoderParamSpec.KERNEL, "must be positive"),  # negative element in 2d
        ((1, 0), EncoderParamSpec.KERNEL, "must be positive"),  # zero element in 2d
        ((-1, 3, 3), EncoderParamSpec.KERNEL, "must be positive"),  # negative element in 3d
    ]
)
def test_validate_param_invalid(value: IntOrTuple2d3d, name: EncoderParamSpec, exc_msg: str) -> None:
    """
    Test that various invalid inputs raise an error once passed to _validate_param
    """
    with pytest.raises(ValueError, match=exc_msg):
        _validate_param(value, name)

# --------------------------- Tests for EncoderParams --------------------------------

def assert_convparams_invalid(exc_msg: str, **kwargs: Any) -> None:
    with pytest.raises(ValueError, match=exc_msg):
        EncoderParams.from_inputs(**kwargs, num_dims=2)

@pytest.mark.parametrize(
    "value,exc_msg",
    [
        (-1, "must be positive"),       # negative int
        (0, "must be positive"),      # zero int
        ((1,), "must have length 2 or 3"),  # tuple too short
        ((1, 1, 1, 1), "must have length 2 or 3"),  # tuple too long
        ((1, "a"), "must be int"),       # non-int element
        ((1, -1), "must be positive"),  # negative element in 2d
        ((1, 0), "must be positive"),  # zero element in 2d
        ((-1, 3, 3), "must be positive"),  # negative element in 3d
    ]
)
def test_assert_convparams_invalid(value: IntOrTuple2d3d, exc_msg: str) -> None:
    """
    Test that various invalid inputs raise an error once passed to EncoderParams.from_inputs
    """
    assert_convparams_invalid(exc_msg=exc_msg, kernel=value, dilation=1, stride=1)
    assert_convparams_invalid(exc_msg=exc_msg, kernel=1, dilation=value, stride=1)
    assert_convparams_invalid(exc_msg=exc_msg, kernel=1, dilation=1, stride=value)


@pytest.mark.parametrize(
    "value,exc_msg",
    [
        (2, "Even kernel sizes cannot preserve spatial dimensions"),
        (4, "Even kernel sizes cannot preserve spatial dimensions"),
    ]
)
def test_assert_kernel_invalid(value: IntOrTuple2d3d, exc_msg: str) -> None:
    """
    Test that even kernels raise an error once passed to EncoderParams.from_inputs
    """
    with pytest.raises(ValueError, match=exc_msg):
        EncoderParams.from_inputs(kernel=value, dilation=1, stride=1, num_dims=2)


def test_from_inputs_scalar_2d() -> None:
    params = EncoderParams.from_inputs(kernel=3,
                                       dilation=1,
                                       stride=1,
                                       num_dims=2)

    assert params.kernel == (3, 3)
    assert params.dilation == (1, 1)
    assert params.stride == (1, 1)

def test_from_inputs_tuples_2d_1() -> None:
    params = EncoderParams.from_inputs(kernel=(3, 3),
                                       dilation=1,
                                       stride=1)

    assert params.kernel == (3, 3)
    assert params.dilation == (1, 1)
    assert params.stride == (1, 1)

def test_from_inputs_tuples_2d_2() -> None:
    params = EncoderParams.from_inputs(kernel=(3, 3),
                                       dilation=(1, 1),
                                       stride=(1, 1),
                                       num_dims=2)

    assert params.kernel == (3, 3)
    assert params.dilation == (1, 1)
    assert params.stride == (1, 1)

def test_from_inputs_scalar_3d() -> None:
    params = EncoderParams.from_inputs(kernel=3,
                                       dilation=1,
                                       stride=2,
                                       num_dims=3)

    assert params.kernel == (3, 3, 3)
    assert params.dilation == (1, 1, 1)
    assert params.stride == (2, 2, 2)

def test_from_inputs_tuples_3d() -> None:
    params = EncoderParams.from_inputs(kernel=(1, 3, 3),
                                       dilation=(1, 3, 3),
                                       stride=(1, 2, 2),
                                       num_dims=3)

    assert params.kernel == (1, 3, 3)
    assert params.dilation == (1, 3, 3)
    assert params.stride == (1, 2, 2)

# ------------------- Test padding computation -------------------

@pytest.mark.parametrize(
    "kernel,dilation,expected_padding",
    [
        ((3, 3), (1, 1), (1, 1)),  # k=2n+1, d=2n+1
        ((5, 5), (1, 1), (2, 2)),
        ((3, 3), (2, 2), (2, 2)),  # k=2n+1, d=2n
        ((5, 5), (2, 2), (4, 4)),
        ((3, 5), (2, 3), (2, 6)),  # mixed
        ((3, 3, 3), (1, 1, 1), (1, 1, 1)),
    ]
)
def test_padding_auto(kernel: Union[TupleOfInt2d, TupleOfInt3d],
                      dilation: Union[TupleOfInt2d, TupleOfInt3d],
                      expected_padding: Union[TupleOfInt2d, TupleOfInt3d]) -> None:
    """
    Test the value of computed padding in EncoderParams
    """
    for s in (1, 2):
        params = EncoderParams.from_inputs(kernel=kernel, dilation=dilation, stride=s)
        assert params.padding == expected_padding

# ------------------- Test as_2d -------------------

def test_as_2d_equivalency_to_convparams() -> None:
    params = EncoderParams.from_inputs(kernel=(3, 3), dilation=(1, 1), stride=1)
    params2d = params.as_2d()
    assert params2d.padding == params.padding
    assert params2d.kernel == params.kernel

@pytest.mark.parametrize(
    "kernel,stride,dilation, exc_msg",
    [((3, 3, 3), (1, 1), (1, 1), "must be 2D"),
     ((3, 3), (1, 1, 1), (1, 1), "must be 2D"),
     ((3, 3), (1, 1), (1, 1, 1), "must be 2D")],
)
def test_as_2d_error_for_3d(kernel: IntOrTuple2d, stride: IntOrTuple2d, dilation: IntOrTuple2d, exc_msg: str) -> None:
    """
    Test get_encoder_params_2d raises an error if provided tuples are not 2d
    """
    with pytest.raises(ValueError, match=exc_msg):
        _ = get_encoder_params_2d(kernel=kernel, dilation=dilation, stride=stride)

# ------------------- Test helper function get_encoder_params_2d -------------------
@pytest.mark.parametrize(
    "kernel, stride, dilation, expected_kernel, expected_stride, expected_dilation, expected_padding",
    [
        (3, 1, 1, (3, 3), (1, 1), (1, 1), (1, 1)),  # k=2n+1, d=2n+1;
        (3, 2, 1, (3, 3), (2, 2), (1, 1), (1, 1)),
        ((3, 3), (1, 1), (1, 1), (3, 3), (1, 1), (1, 1), (1, 1)),
        (3, 1, 2, (3, 3), (1, 1), (2, 2), (2, 2)),  # k=2n+1, d=2n;
        (3, 2, 2, (3, 3), (2, 2), (2, 2), (2, 2))
    ]
)
def test_get_encoder_params_2d_returns(kernel: IntOrTuple2d, stride: IntOrTuple2d, dilation: IntOrTuple2d,
                                       expected_kernel: TupleOfInt2d, expected_stride: TupleOfInt2d,
                                       expected_dilation: TupleOfInt2d, expected_padding: TupleOfInt2d) -> None:
    """
    Test the values of instance variables of EncoderParams returned by get_encoder_params_2d
    """
    params2d = get_encoder_params_2d(kernel=kernel, stride=stride, dilation=dilation)
    assert params2d.padding == expected_padding
    assert params2d.kernel == expected_kernel
    assert params2d.stride == expected_stride
    assert params2d.dilation == expected_dilation

@pytest.mark.parametrize(
    "kernel, stride, dilation",
    [
        (4, 1, 1),
        (4, 2, 1)
    ]
)
def test_get_encoder_params_2d_even_kernels_raises_error(kernel: IntOrTuple2d, stride: IntOrTuple2d, dilation: IntOrTuple2d) -> None:
    """
    Test that even kernel sizes raise an error
    """
    with pytest.raises(ValueError, match="Even kernel sizes cannot preserve spatial dimensions"):
        _ = get_encoder_params_2d(kernel=kernel, stride=stride, dilation=dilation)
