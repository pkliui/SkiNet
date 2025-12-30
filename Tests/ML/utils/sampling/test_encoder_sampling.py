import pytest

from SkiNet.ML.utils.sampling.encoder_sampling import (ConvParams, ConvParamSpec, _normalize_kernel_dilation,
                                                       _validate_param, get_padding)
from SkiNet.ML.utils.typing_utils import IntOrTuple2d3d, TupleOfInt2d3d

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

# --------------------------- Tests for _ensure_integer_padding --------------------

@pytest.mark.parametrize(
    "kernel,dilation,stride",
    [
        # invalid cases (should raise)
        ((4, 4), (1, 1), 1),  # even kernel , odd dilation
        ((3, 4), (1, 1), 1),
        ((4, 4), (1, 1), 2),
        ((3, 4), (1, 1), 2),

    ]
)
def test_ensure_integer_padding_error(kernel: IntOrTuple2d3d, dilation: IntOrTuple2d3d, stride: IntOrTuple2d3d) -> None:
    with pytest.raises(ValueError):
        params = ConvParams.from_inputs(kernel=kernel, dilation=dilation, stride=stride)
        params._ensure_integer_padding()

# --------------------------- Tests for _validate_param --------------------------

@pytest.mark.parametrize(
    "value,name",
    [
        (1, ConvParamSpec.KERNEL),
        ((2, 2), ConvParamSpec.KERNEL),
        ((3, 3), ConvParamSpec.KERNEL),
        ((1, 2), ConvParamSpec.KERNEL),
        ((1, 2, 2), ConvParamSpec.KERNEL),
        ((1, 2, 3), ConvParamSpec.KERNEL),
        (1, ConvParamSpec.DILATION),
        ((1, 1), ConvParamSpec.DILATION),
        ((2, 2), ConvParamSpec.DILATION),
        ((1, 2, 2), ConvParamSpec.DILATION),
        ((1, 2, 3), ConvParamSpec.DILATION),
        (1, ConvParamSpec.STRIDE),
        ((1, 1), ConvParamSpec.STRIDE),
        ((2, 2), ConvParamSpec.STRIDE),
        ((1, 2, 2), ConvParamSpec.STRIDE)
    ]
)
def test_validate_param_valid(value: IntOrTuple2d3d, name: ConvParamSpec) -> None:
    assert _validate_param(value, name) == value


@pytest.mark.parametrize(
    "value,name,exc_msg",
    [
        (-1, ConvParamSpec.KERNEL, "must be positive"),       # negative int
        (0, ConvParamSpec.DILATION, "must be positive"),      # zero int
        ((1,), ConvParamSpec.KERNEL, "must have length 2 or 3"),  # tuple too short
        ((1, 2, 3, 4), ConvParamSpec.DILATION, "must have length 2 or 3"),  # tuple too long
        ((1, "a"), ConvParamSpec.STRIDE, "must be int"),       # non-int element
        ((1, -2), ConvParamSpec.KERNEL, "must be positive"),  # negative element in 2d
        ((1, 0), ConvParamSpec.KERNEL, "must be positive"),  # zero element in 2d
        ((-1, 2, 2), ConvParamSpec.KERNEL, "must be positive"),  # negative element in 3d
    ]
)
def test_validate_param_invalid(value: IntOrTuple2d3d, name: ConvParamSpec, exc_msg: str) -> None:
    with pytest.raises(ValueError, match=exc_msg):
        _validate_param(value, name)


# --------------------------- Tests for ConvParams --------------------------------
def test_from_inputs_scalar_2d() -> None:
    params = ConvParams.from_inputs(
        kernel=3,
        dilation=1,
        stride=1,
        num_dims=2,
    )

    assert params.kernel == (3, 3)
    assert params.dilation == (1, 1)
    assert params.stride == 1

def test_from_inputs_tuples_2d_1() -> None:
    params = ConvParams.from_inputs(
        kernel=(3, 3),
        dilation=(1, 1),
        stride=1,
        num_dims=2,
    )

    assert params.kernel == (3, 3)
    assert params.dilation == (1, 1)
    assert params.stride == 1

def test_from_inputs_tuples_2d_2() -> None:
    params = ConvParams.from_inputs(
        kernel=(3, 3),
        dilation=(1, 1),
        stride=(1, 1),
        num_dims=2,
    )

    assert params.kernel == (3, 3)
    assert params.dilation == (1, 1)
    assert params.stride == (1, 1)

def test_from_inputs_scalar_3d() -> None:
    params = ConvParams.from_inputs(
        kernel=3,
        dilation=1,
        stride=2,
        num_dims=3,
    )

    assert params.kernel == (3, 3, 3)
    assert params.dilation == (1, 1, 1)
    assert params.stride == 2


def test_from_inputs_tuples_3d() -> None:
    params = ConvParams.from_inputs(
        kernel=(1, 3, 3),
        dilation=(1, 3, 3),
        stride=(1, 2, 2),
        num_dims=3,
    )

    assert params.kernel == (1, 3, 3)
    assert params.dilation == (1, 3, 3)
    assert params.stride == (1, 2, 2)

# ----------------------------  Tests for get_padding  ----------------------------

class TestGetPadding:
    """Test cases for get_padding function"""

    @pytest.mark.parametrize("kernel, dilation, stride, expected_padding", [
        # even kernels, even dilations result in integer padding
        # odd kernels, odd dilations result in integer padding
        # odd kernels, even dilations result in integer padding
        ((2, 2), (2, 2), 1, (1, 1)),   # kernel=2, dilation=2 -> padding=1
        ((3, 3), (1, 1), 1, (1, 1)),   # kernel=3, dilation=1 -> padding=1
        ((3, 3), (2, 2), 1, (2, 2)),   # kernel=3, dilation=2 -> padding=2

    ])
    def test_get_padding_success(self,
                                 kernel: TupleOfInt2d3d,
                                 dilation: TupleOfInt2d3d,
                                 stride: TupleOfInt2d3d,
                                 expected_padding: TupleOfInt2d3d) -> None:
        """
        Test get_padding, where input parameters result in integer padding
        """
        params = ConvParams.from_inputs(
            kernel=kernel,
            dilation=dilation,
            stride=stride,
            num_dims=2,
        )
        padding = get_padding(params)
        assert len(padding) == 2
        assert padding == expected_padding
        assert padding == expected_padding
