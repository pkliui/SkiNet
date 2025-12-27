from typing import Union, cast

import pytest

from SkiNet.ML.utils.sampling.encoder_sampling import (PaddingMode, _compute_stride2_padding, _normalize_conv_inputs,
                                                       get_padding)
from SkiNet.ML.utils.typing_utils import IntOrTuple2d3d, TupleOfInt2d3d


# ---------------------------------- Tests for PaddingMode --------------------------------------#
class TestPaddingModeEnum:

    def test_padding_mode_properties_and_spec_payload(self) -> None:
        """Test that PaddingMode enum members have correct properties and payload."""

        assert PaddingMode.SAME.value.value_str == "same"
        assert PaddingMode.SAME.downsampling_factor == 1
        assert PaddingMode.SAME.stride == 1

        assert PaddingMode.DOWNSAMPLING_FACTOR_2.value.value_str == "downsampling_factor_2"
        assert PaddingMode.DOWNSAMPLING_FACTOR_2.downsampling_factor == 2
        assert PaddingMode.DOWNSAMPLING_FACTOR_2.stride == 2

    def test_padding_mode_from_value(self) -> None:
        """Test that from_value returns the same enum member when given a valid PaddingMode."""
        assert PaddingMode.from_value(PaddingMode.DOWNSAMPLING_FACTOR_2) is PaddingMode.DOWNSAMPLING_FACTOR_2

    def test_padding_mode_from_value_invalid_raises(self) -> None:
        """Test that from_value raises an error for non-enum inputs (strict API)."""
        with pytest.raises(ValueError, match="is not a valid"):
            PaddingMode.from_value(cast(PaddingMode, "not_a_mode"))  # cast for mypy
        with pytest.raises(ValueError, match="is not a valid"):
            PaddingMode.from_value(cast(PaddingMode, 123))
        with pytest.raises(ValueError, match="is not a valid"):
            PaddingMode.from_value(cast(PaddingMode, None))
        with pytest.raises(ValueError, match="is not a valid"):
            PaddingMode.from_value(cast(PaddingMode, "same"))
        with pytest.raises(ValueError, match="is not a valid"):
            PaddingMode.from_value(cast(PaddingMode, "SAME"))

# ---------------------------- Tests for _normalize_conv_inputs----------------------------

class TestHandleMixedInputs:
    """Test cases for _normalize_conv_inputs function"""

    def test_all_scalars_uses_num_dims(self) -> None:
        """Test that when all inputs are scalars, num_dims is used"""
        params = _normalize_conv_inputs(3, 1, 3)

        assert params.kernel == (3, 3, 3)
        assert params.dilation == (1, 1, 1)

    def test_all_scalars_default_num_dims(self) -> None:
        """Test that when all inputs are scalars and num_dims = 2 is used"""
        params = _normalize_conv_inputs(5, 2, 2)

        assert params.kernel == (5, 5)
        assert params.dilation == (2, 2)

    def test_mixed_inputs_uses_max_length(self) -> None:
        """Test that when inputs are mixed, max length is used"""
        params = _normalize_conv_inputs((3, 5, 7), 1)

        assert params.kernel == (3, 5, 7)
        assert params.dilation == (1, 1, 1)

    def test_scalar_with_iterable(self) -> None:
        """Test scalar with iterable input"""
        params = _normalize_conv_inputs(3, (1, 2, 3))

        assert params.kernel == (3, 3, 3)
        assert params.dilation == (1, 2, 3)

    def test_iterable_with_scalar(self) -> None:
        """Test iterable with scalar input"""
        params = _normalize_conv_inputs((3, 5, 7), 2)

        assert params.kernel == (3, 5, 7)
        assert params.dilation == (2, 2, 2)

    def test_zero_values(self) -> None:
        """Test with zero values - should raise ValueError"""
        with pytest.raises(ValueError, match="kernel must be positive, got 0"):
            _normalize_conv_inputs(0, (1, 1))

    def test_num_dims_ignored_when_iterables_present(self) -> None:
        """Test that num_dims is ignored when iterables are present"""
        params = _normalize_conv_inputs((3, 5), 1, 3)

        # Should use max length (2) from the iterable, not num_dims (3)!
        assert params.kernel == (3, 5)
        assert params.dilation == (1, 1)
        assert len(params.kernel) == 2
        assert len(params.dilation) == 2

    @pytest.mark.parametrize(
        "kernel, dilation, error_msg",
        [
            ((3, 5, 7), (1, 2), "All inputs must have the same length"),
            ((3, 5), (1, 1, 1), "All inputs must have the same length"),
        ]
    )
    def test_tuple_inputs(self, kernel: IntOrTuple2d3d, dilation: IntOrTuple2d3d, error_msg: str) -> None:
        """Test with inputs of different size - should raise ValueError for different sizes"""
        with pytest.raises(ValueError, match=error_msg):
            _normalize_conv_inputs(kernel, dilation)

    @pytest.mark.parametrize(
        "kernel, dilation, error_msg",
        [
            (3.5, 1, "must contain only integers or tuples of integers"),
            (3, 1.5, "must contain only integers or tuples of integers"),
            ((3, 2.5), (1, 1), "must contain only integers"),
            ((3, 5), (1, 1.5), "must contain only integers"),
        ]
    )
    def test_float_type_errors(self, kernel: IntOrTuple2d3d, dilation: IntOrTuple2d3d, error_msg: str) -> None:
        """Test that float values in kernel/dilation raise TypeError"""
        with pytest.raises(TypeError, match=error_msg):
            _normalize_conv_inputs(kernel, dilation)

    @pytest.mark.parametrize(
        "kernel, dilation, error_msg",
        [
            ("3", 1, "must contain only integers or tuples of integers"),
            (3, "1", "must contain only integers or tuples of integers"),
            ((3, "5"), (1, 1), "must contain only integers"),
            ((3, 5), (1, "1"), "must contain only integers"),
        ]
    )
    def test_string_type_errors(self, kernel: IntOrTuple2d3d, dilation: IntOrTuple2d3d, error_msg: str) -> None:
        """Test that string values in kernel/dilation raise TypeError"""
        with pytest.raises(TypeError, match=error_msg):
            _normalize_conv_inputs(kernel, dilation)

# ----------------------------  Tests for DOWNSAMPLING_FACTOR_2 sampling mode with stride=2 ----------------------------

class TestGetPaddingStride2:
    """Test cases for _compute_stride2_padding function"""

    @pytest.mark.parametrize("kernel, dilation, expected_padding", [
        # even kernels, odd dilations result in integer padding
        ((2, 2), (1, 1), 0),   # kernel=2, dilation=1 -> padding=0
        ((4, 4), (1, 1), 1),   # kernel=4, dilation=1 -> padding=1
        ((2, 2), (3, 3), 1),   # kernel=2, dilation=3 -> padding=2
        ((4, 4), (3, 3), 4),   # kernel=4, dilation=3 -> padding=4
    ])
    def test_compute_stride2_padding_success(self,
                                             kernel: TupleOfInt2d3d,
                                             dilation: TupleOfInt2d3d,
                                             expected_padding: Union[TupleOfInt2d3d, str]) -> None:
        """Test get_padding_stride_2, where input parameters result in integer padding
        """
        padding = _compute_stride2_padding(kernel, dilation)
        assert len(padding) == 2  # default
        assert padding[0] == expected_padding
        assert padding[1] == expected_padding

    @pytest.mark.parametrize("kernel, dilation, expected_padding", [
        # even kernels, odd dilations result in integer padding
        ((2, 2, 2), (1, 1, 1), 0),   # kernel=2, dilation=1 -> padding=0
        ((4, 4, 4), (3, 3, 3), 4),   # kernel=4, dilation=3 -> padding=4
    ])
    def test_compute_stride2_padding_success_3d(self,
                                                kernel: TupleOfInt2d3d,
                                                dilation: TupleOfInt2d3d,
                                                expected_padding: Union[TupleOfInt2d3d, str]) -> None:
        """Test get_padding_stride_2, where input parameters result in integer padding, 3d case
        """
        padding = _compute_stride2_padding(kernel, dilation)
        assert len(padding) == 3  # default
        assert padding[0] == expected_padding
        assert padding[1] == expected_padding
        assert padding[2] == expected_padding

# ---------------------------- Tests for  get_padding function ----------------------------

class TestGetPaddingAndStride:
    """Test cases for get_padding function"""

    @pytest.mark.parametrize("kernel, dilation, expected_padding, padding_mode", [
        (2, 2, 'same', PaddingMode.SAME),  # scalar inputs are expanded with correct number of dimensions to default num=2
        ((3, 3), (1, 1), 'same', PaddingMode.SAME),  # kernel and dilation have the same size, they are passed through without modification
        ((3, 3), 1, 'same', PaddingMode.SAME),  # when kernel is iterable and dilation is scalar, dilation is expanded to match kernel size
        (2, 1, 0, PaddingMode.DOWNSAMPLING_FACTOR_2),  # scalar inputs are expanded with correct number of dimensions to default num=2
        # when both kernel and dilation are iterables of the same size, they are passed through without modification
        ((4, 4), (1, 1), 1, PaddingMode.DOWNSAMPLING_FACTOR_2),
        ((2, 2), 1, 0, PaddingMode.DOWNSAMPLING_FACTOR_2),  # when kernel is iterable and dilation is scalar, dilation is expanded to match kernel size

    ])
    def test_get_padding_integration_with__normalize_conv_inputs_success(self, kernel: IntOrTuple2d3d,
                                                                         dilation: IntOrTuple2d3d,
                                                                         expected_padding: Union[TupleOfInt2d3d, str],
                                                                         padding_mode: PaddingMode) -> None:
        """
        Integration with _normalize_conv_inputs for success cases
        """
        padding = get_padding(kernel, dilation, padding_mode)
        assert padding == 'same' if padding_mode == PaddingMode.SAME else (expected_padding, expected_padding)  # we have num_dims=2 by default

    @pytest.mark.parametrize("kernel, dilation, padding_mode", [
        ((3, 3), (1, 1, 1), PaddingMode.SAME),  # a ValueError is raised when kernel and dilation are iterables of different sizes
        ((3, 3, 3), (3, 3), PaddingMode.DOWNSAMPLING_FACTOR_2),
    ])
    def test_get_padding_integration_with__normalize_conv_inputs_diff_sizes(self, kernel: IntOrTuple2d3d,
                                                                            dilation: IntOrTuple2d3d, padding_mode: PaddingMode) -> None:
        """Integration with _normalize_conv_inputs for error cases - different sizes"""
        with pytest.raises(ValueError, match="All inputs must have the same length"):
            get_padding(kernel, dilation, padding_mode)

    @pytest.mark.parametrize("kernel, dilation, padding_mode, error_msg", [
        ((3, 2.5), (1, 1), PaddingMode.SAME, "must contain only integers"),
        ((3, "5"), (1, 1), PaddingMode.SAME, "must contain only integers"),
        ((3, 5), (1, 1.5), PaddingMode.SAME, "must contain only integers"),
        ((3, 5), (1, "1"), PaddingMode.SAME, "must contain only integers"),
        ((3, 2.5), (1, 1), PaddingMode.DOWNSAMPLING_FACTOR_2, "must contain only integers"),
        ((3, "5"), (1, 1), PaddingMode.DOWNSAMPLING_FACTOR_2, "must contain only integers"),
        ((3, 5), (1, 1.5), PaddingMode.DOWNSAMPLING_FACTOR_2, "must contain only integers"),
        ((3, 5), (1, "1"), PaddingMode.DOWNSAMPLING_FACTOR_2, "must contain only integers"),
    ])
    def test_get_padding_integration_with__normalize_conv_inputs_invalid_types(self, kernel: IntOrTuple2d3d,
                                                                               dilation: IntOrTuple2d3d,
                                                                               padding_mode: PaddingMode,
                                                                               error_msg: str) -> None:
        """Integration with _normalize_conv_inputs for error cases - invalid types"""
        with pytest.raises(TypeError, match=error_msg):
            get_padding(kernel, dilation, padding_mode)

    @pytest.mark.parametrize("sampling_mode", [
        "invalid_mode",
        "DOWNSAMPLING_FACTOR_22",
    ])
    def test_wrong_sampling_mode(self, sampling_mode: PaddingMode) -> None:
        """Test get_padding_value_and_stride, when sampling mode is not in PaddingMode"""
        with pytest.raises(ValueError, match="is not a valid"):  # propagated from _ensure_padding_mode
            _ = get_padding(1, 3, sampling_mode)

    @pytest.mark.parametrize(
        "padding_mode,kernel,dilation,expected_padding", [
            # SAME mode cases (stride forced to 1)
            # odd kernels, odd dilation result in integer padding
            (PaddingMode.SAME, 3, 1, 'same'),  # kernel=3, dilation=1 -> padding=1
            # even kernels, even dilation result in integer padding
            (PaddingMode.SAME, 4, 2, 'same'),  # kernel=4, dilation=2 -> padding=3
            # odd kernels, even dilation result in integer padding
            (PaddingMode.SAME, 5, 2, 'same'),  # kernel=5, dilation=2 -> padding=4
            # DOWNSAMPLING_FACTOR_2 mode cases (stride forced to 2)
            # even kernels, odd dilations result in integer padding
            (PaddingMode.DOWNSAMPLING_FACTOR_2, 2, 1, (0, 0)),   # kernel=2, dilation=1 -> padding=0
        ])
    def test_padding_modes_unified(self, padding_mode: PaddingMode,
                                   kernel: IntOrTuple2d3d,
                                   dilation: IntOrTuple2d3d,
                                   expected_padding: Union[TupleOfInt2d3d, str]) -> None:
        """
        Unified test covering SAME, and DOWNSAMPLING_FACTOR_2 modes for the default value of num_dims=2.
        """
        padding = get_padding(kernel=kernel, dilation=dilation, padding_mode=padding_mode)
        assert padding == expected_padding

    def test_get_padding_invalid_mode_raises(self) -> None:
        """
        Test get_padding raises ValueError for invalid padding mode string raised through _ensure_padding_mode
        """
        with pytest.raises(ValueError):
            get_padding(kernel=3, dilation=1, padding_mode=cast(PaddingMode, "not_a_mode"))  # cast for mypy only
            get_padding(kernel=3, dilation=1, padding_mode=cast(PaddingMode, "not_a_mode"))  # cast for mypy only
