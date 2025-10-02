import pytest
from SkiNet.ML.utils.sampling.encoder_sampling import PaddingMode, get_padding_stride_2, get_padding_and_stride, get_same_padding_stride_1, handle_mixed_inputs

    ######################################## Tests for handle_mixed_inputs ########################################

class TestHandleMixedInputs:
    """Test cases for handle_mixed_inputs function"""

    def test_all_scalars_uses_num_dims(self):
        """Test that when all inputs are scalars, num_dims is used"""
        kernel, dilation = handle_mixed_inputs(3, 1, num_dims=4)
        
        assert kernel == [3, 3, 3, 3]
        assert dilation == [1, 1, 1, 1]

    def test_all_scalars_default_num_dims(self):
        """Test that when all inputs are scalars, default num_dims=2 is used"""
        kernel, dilation = handle_mixed_inputs(5, 2)
        
        assert kernel == [5, 5]
        assert dilation == [2, 2]

    def test_mixed_inputs_uses_max_length(self):
        """Test that when inputs are mixed, max length is used"""
        kernel, dilation = handle_mixed_inputs([3, 5, 7], 1)
        
        assert kernel == [3, 5, 7]
        assert dilation == [1, 1, 1]

    def test_single_element_iterables(self):
        """Test with single-element iterables"""
        kernel, dilation = handle_mixed_inputs([3], (1,))
        
        assert kernel == [3]
        assert dilation == [1]

    def test_scalar_with_iterable(self):
        """Test scalar with iterable input"""
        kernel, dilation = handle_mixed_inputs(3, [1, 2, 3])
        
        assert kernel == [3, 3, 3]
        assert dilation == [1, 2, 3]

    def test_iterable_with_scalar(self):
        """Test iterable with scalar input"""
        kernel, dilation = handle_mixed_inputs([3, 5, 7], 2)
        
        assert kernel == [3, 5, 7]
        assert dilation == [2, 2, 2]

    def test_empty_iterables(self):
        """Test with empty iterables - should raise ValueError"""
        with pytest.raises(ValueError, match="Empty kernel is not allowed"):
            handle_mixed_inputs([], [])

    def test_zero_values(self):
        """Test with zero values - should raise ValueError"""
        with pytest.raises(ValueError, match="kernel must be positive, got 0"):
            handle_mixed_inputs(0, [1, 1])

    def test_num_dims_ignored_when_iterables_present(self):
        """Test that num_dims is ignored when iterables are present"""
        kernel, dilation = handle_mixed_inputs([3, 5], 1, num_dims=10)
        
        # Should use max length (2) from the iterable, not num_dims (10)
        assert kernel == [3, 5]
        assert dilation == [1, 1]
        assert len(kernel) == 2
        assert len(dilation) == 2

    @pytest.mark.parametrize(
        "kernel, dilation, error_msg",
        [
            ((3, 5, 7), (1, 2), "All iterable inputs must have the same size"),
            ([3, 5], [1, 2, 3], "All iterable inputs must have the same size"),
            ((3, 5), (1,), "All iterable inputs must have the same size"),
            ([3,], [1, 2], "All iterable inputs must have the same size")
        ]
    )
    def test_tuple_inputs(self, kernel, dilation, error_msg):
        """Test with inputs of different size - should raise ValueError for different sizes"""
        with pytest.raises(ValueError, match=error_msg):
            handle_mixed_inputs(kernel, dilation)

    @pytest.mark.parametrize(
        "kernel, dilation, error_msg",
        [
            (3.5, 1, "kernel must be an integer"),
            (3, 1.5, "dilation must be an integer"),
            ([3, 2.5], [1, 1], "kernel must contain only integers"),
            ([3, 5], [1, 1.5], "dilation must contain only integers"),
        ]
    )
    def test_float_type_errors(self, kernel, dilation, error_msg):
        """Test that float values in kernel/dilation raise TypeError"""
        with pytest.raises(TypeError, match=error_msg):
            handle_mixed_inputs(kernel, dilation)

    @pytest.mark.parametrize(
        "kernel, dilation, error_msg",
        [
            ("3", 1, "kernel must be an integer"),
            (3, "1", "dilation must be an integer"),
            ([3, "5"], [1, 1], "kernel must contain only integers"),
            ([3, 5], [1, "1"], "dilation must contain only integers"),
        ]
    )
    def test_string_type_errors(self, kernel, dilation, error_msg):
        """Test that string values in kernel/dilation raise TypeError"""
        with pytest.raises(TypeError, match=error_msg):
            handle_mixed_inputs(kernel, dilation)

    ######################################## Tests for SAME padding mode with stride=1 ########################################

class TestSamePaddingStride1:
    """Test cases for get_same_padding_stride_1 function"""

    @pytest.mark.parametrize("kernel, dilation, expected_padding", [
        (2, 2, 1), # scalar inputs are expanded with correct number of dimensions to default num=2
        ([3, 3], [1, 1], 1),  # when both kernel and dilation are iterables of the same size, they are passed through without modification
        ([3, 3], 1, 1),  # when kernel is iterable and dilation is scalar, dilation is expanded to match kernel size
    ])
    def test_get_same_padding_stride_1_integration_with_handle_mixed_inputs_success(self, kernel, dilation, expected_padding):
        """
        Integration with handle_mixed_inputs for success cases
        """
        padding = get_same_padding_stride_1(kernel, dilation)
        assert len(padding) == 2  # Default num_dims=2 
        assert padding[0] == expected_padding
        assert padding[1] == expected_padding

    @pytest.mark.parametrize("kernel, dilation", [
        ([3, 3], [1]),  # a ValueError is raised when kernel and dilation are iterables of different sizes
    ])
    def test_get_same_padding_stride_1_integration_with_handle_mixed_inputs_error(self, kernel, dilation):
        """
        Integration with handle_mixed_inputs for error cases
        """
        with pytest.raises(ValueError, match="All iterable inputs must have the same size"):
            _ = get_same_padding_stride_1(kernel, dilation)

    @pytest.mark.parametrize("kernel, dilation", [
        # even kernels, odd dilation result in fractional padding
        (2, 1),   # kernel=2, dilation=1 -> padding=1/2
        (6, 1),   # kernel=6, dilation=1 -> padding=5/2
        (2, 3),   # kernel=2, dilation=3 -> padding=3/2
        (6, 3),   # kernel=6, dilation=3 -> padding=15/2
    ])
    def test_get_same_padding_stride_1_error(self, kernel, dilation):
        """
        Test get_same_padding_stride_1, non-integer error cases
        """        
        with pytest.raises(ValueError, match="not all items in padding"):
            _ = get_same_padding_stride_1(kernel, dilation)

    @pytest.mark.parametrize("kernel, dilation, expected_padding", [
        # odd kernels, odd dilation result in integer padding
        (3, 1, 1),   # kernel=3, dilation=1 -> padding=1
        (5, 1, 2),   # kernel=5, dilation=1 -> padding=2
        (3, 3, 3),   # kernel=3, dilation=3 -> padding=3
        (5, 3, 6),   # kernel=5, dilation=3 -> padding=6
        # even kernels, even dilation result in integer padding
        (2, 2, 1),   # kernel=2, dilation=2 -> padding=1
        (4, 2, 3),   # kernel=4, dilation=2 -> padding=3
        (2, 4, 2),   # kernel=2, dilation=4 -> padding=3
        (4, 4, 6),   # kernel=4, dilation=4 -> padding=6
        # odd kernels, even dilation result in integer padding
        (3, 2, 2),   # kernel=3, dilation=2 -> padding=2
        (5, 2, 4),   # kernel=5, dilation=2 -> padding=4
        (3, 4, 4),   # kernel=3, dilation=4 -> padding=6
        (5, 4, 8),  # kernel=5, dilation=4 -> padding=10
    ])
    def test_get_same_padding_stride_1_success(self, kernel, dilation, expected_padding):
        """Test get_same_padding_stride_1, where input parameters result in integer padding
        """
        padding = get_same_padding_stride_1(kernel, dilation)
        assert len(padding) == 2 # default
        assert padding[0] == expected_padding
        assert padding[1] == expected_padding

    @pytest.mark.parametrize("kernel,dilation", [
        # Odd kernels with odd dilation
        (3, 1), (5, 1),
        # Odd kernels with even dilation
        (3, 2), (5, 2),
        # Even kernels with even dilation
        (2, 2), (4, 2),
    ])
    def test_get_same_padding_stride_1_integration_with_get_padding_and_stride(self, kernel, dilation):
        """
        Test get_padding_and_stride (SAME) equals get_same_padding_stride_1 across all valid integer padding cases
        """
        params = get_padding_and_stride(kernel, dilation, PaddingMode.SAME)
        padding_from_get_padding_and_stride = params.padding
        padding_from_get_same_padding_stride_1 = get_same_padding_stride_1(kernel, dilation)

        assert padding_from_get_padding_and_stride == padding_from_get_same_padding_stride_1, (
            f"kernel={kernel}, dilation={dilation}: get_padding_and_stride returned {padding_from_get_padding_and_stride}, "
            f"get_same_padding_stride_1 returned {padding_from_get_same_padding_stride_1}")

    ######################################## Tests for DOWNSAMPLING_FACTOR_2 sampling mode with stride=2 ########################################
        
class TestGetPaddingStride2:
    """Test cases for get_padding_stride_2 function"""

    @pytest.mark.parametrize("kernel, dilation, expected_padding", [
        (2, 1, 0),  # scalar inputs are expanded with correct number of dimensions to default num=2
        ([4, 4], [1, 1], 1), # when both kernel and dilation are iterables of the same size, they are passed through without modification
        ([2, 2], 1, 0),  # when kernel is iterable and dilation is scalar, dilation is expanded to match kernel size

    ])
    def test_get_same_padding_stride_2_integration_with_handle_mixed_inputs_success(self, kernel, dilation, expected_padding):
        """
        Integration with handle_mixed_inputs for success cases
        """
        padding = get_padding_stride_2(kernel, dilation)
        assert len(padding) == 2  # Default num_dims=2 
        assert padding[0] == expected_padding
        assert padding[1] == expected_padding

    @pytest.mark.parametrize("kernel, dilation", [
        ([3, 3], [1]),  # a ValueError is raised when kernel and dilation are iterables of different sizes
    ])
    def test_get_same_padding_stride_2_integration_with_handle_mixed_inputs_error(self, kernel, dilation):
        """
        Integration with handle_mixed_inputs for error cases
        """
        with pytest.raises(ValueError, match="All iterable inputs must have the same size"):
            _ = get_padding_stride_2(kernel, dilation)

    @pytest.mark.parametrize("kernel, dilation", [
        # odd kernels, odd dilations result in fractional padding
        (3, 1),  # kernel=3, dilation=1 -> padding=1/2
        (5, 3),  # kernel=5, dilation=3 -> padding=11/2
        # odd kernels, even dilations result in fractional padding
        (3, 2),  # kernel=3, dilation=2 -> padding=3/2
        (5, 4),  # kernel=5, dilation=4 -> padding=15/2
        # even kernels, even dilations result in fractional padding
        (2, 2),   # kernel=2, dilation=2 -> padding=1/2
        (4, 4),   # kernel=4, dilation=4 -> padding=11/2
    ])
    def test_get_padding_stride_2_error(self, kernel, dilation):
        """
        Test get_padding_stride_2, non-integer error cases
        """
        with pytest.raises(ValueError, match="not all items in padding"):
            _ = get_padding_stride_2(kernel, dilation)

    @pytest.mark.parametrize("kernel, dilation, expected_padding", [
        # even kernels, odd dilations result in integer padding
        (2, 1, 0),   # kernel=2, dilation=1 -> padding=0
        (4, 1, 1),   # kernel=4, dilation=1 -> padding=1
        (2, 3, 1),   # kernel=2, dilation=3 -> padding=2
        (4, 3, 4),   # kernel=4, dilation=3 -> padding=4
    ])
    def test_get_padding_stride_2_success(self, kernel, dilation, expected_padding):
        """Test get_padding_stride_2, where input parameters result in integer padding
        """
        padding = get_padding_stride_2(kernel, dilation)
        assert len(padding) == 2 # default
        assert padding[0] == expected_padding
        assert padding[1] == expected_padding

    ######################################## Tests for  get_padding_and_stride function ########################################

class TestGetPaddingAndStride:
    """Test cases for get_padding_and_stride function"""

    @pytest.mark.parametrize("kernel, dilation, expected_padding, padding_mode", [
        (2, 2, 1, PaddingMode.SAME), # scalar inputs are expanded with correct number of dimensions to default num=2
        ([3, 3], [1, 1], 1, PaddingMode.SAME),  # when both kernel and dilation are iterables of the same size, they are passed through without modification
        ([3, 3], 1, 1, PaddingMode.SAME),  # when kernel is iterable and dilation is scalar, dilation is expanded to match kernel size
        (2, 1, 0, PaddingMode.DOWNSAMPLING_FACTOR_2),  # scalar inputs are expanded with correct number of dimensions to default num=2
        ([4, 4], [1, 1], 1, PaddingMode.DOWNSAMPLING_FACTOR_2), # when both kernel and dilation are iterables of the same size, they are passed through without modification
        ([2, 2], 1, 0, PaddingMode.DOWNSAMPLING_FACTOR_2),  # when kernel is iterable and dilation is scalar, dilation is expanded to match kernel size

    ])
    def test_get_padding_and_stride_integration_with_handle_mixed_inputs_success(self, kernel, dilation, expected_padding, padding_mode):
        """
        Integration with handle_mixed_inputs for success cases
        """
        padding = get_padding_and_stride(kernel, dilation, padding_mode)
        assert len(padding.padding) == 2  # Default num_dims=2
        assert padding.padding[0] == expected_padding
        assert padding.padding[1] == expected_padding

    @pytest.mark.parametrize("kernel, dilation, padding_mode", [
        ([3, 3], [1, 1, 1], PaddingMode.SAME),  # a ValueError is raised when kernel and dilation are iterables of different sizes
        ([3,], [1, 1], PaddingMode.VALID),
        ([3, 3], [3,], PaddingMode.DOWNSAMPLING_FACTOR_2),
    ])
    def test_get_padding_and_stride_integration_with_handle_mixed_inputs_diff_sizes(self, kernel, dilation, padding_mode):
        """Integration with handle_mixed_inputs for error cases - different sizes"""
        with pytest.raises(ValueError, match="All iterable inputs must have the same size"):
            if padding_mode == PaddingMode.VALID:
                get_padding_and_stride(kernel, dilation, padding_mode, stride=1)
            else:
                get_padding_and_stride(kernel, dilation, padding_mode)

    @pytest.mark.parametrize("kernel, dilation, padding_mode, error_msg", [
        ([3, 2.5], [1, 1], PaddingMode.SAME, "kernel must contain only integers"),
        ([3, "5"], [1, 1], PaddingMode.SAME, "kernel must contain only integers"),
        ([3, 5], [1, 1.5], PaddingMode.SAME, "dilation must contain only integers"),
        ([3, 5], [1, "1"], PaddingMode.SAME, "dilation must contain only integers"),
        ([3, 2.5], [1, 1], PaddingMode.VALID, "kernel must contain only integers"),
        ([3, "5"], [1, 1], PaddingMode.VALID, "kernel must contain only integers"),
        ([3, 5], [1, 1.5], PaddingMode.VALID, "dilation must contain only integers"),
        ([3, 5], [1, "1"], PaddingMode.VALID, "dilation must contain only integers"),
        ([3, 2.5], [1, 1], PaddingMode.DOWNSAMPLING_FACTOR_2, "kernel must contain only integers"),
        ([3, "5"], [1, 1], PaddingMode.DOWNSAMPLING_FACTOR_2, "kernel must contain only integers"),
        ([3, 5], [1, 1.5], PaddingMode.DOWNSAMPLING_FACTOR_2, "dilation must contain only integers"),
        ([3, 5], [1, "1"], PaddingMode.DOWNSAMPLING_FACTOR_2, "dilation must contain only integers"),
    ])
    def test_get_padding_and_stride_integration_with_handle_mixed_inputs_invalid_types(self, kernel, dilation, padding_mode, error_msg):
        """Integration with handle_mixed_inputs for error cases - invalid types"""
        with pytest.raises(TypeError, match=error_msg):
            if padding_mode == PaddingMode.VALID:
                get_padding_and_stride(kernel, dilation, padding_mode, stride=1)
            else:
                get_padding_and_stride(kernel, dilation, padding_mode)

    @pytest.mark.parametrize("sampling_mode", [
        "invalid_mode",
        "DOWNSAMPLING_FACTOR_22", 
    ])
    def test_wrong_sampling_mode(self, sampling_mode):
        """Test get_padding_value_and_stride, when sampling mode is not in PaddingMode"""
        with pytest.raises(ValueError, match="Invalid sampling mode:"):
            _ = get_padding_and_stride(1, 3, sampling_mode)

    @pytest.mark.parametrize("stride, kernel, dilation", [
        (None, 3, 1),
        (None, 5, 1),
    ])
    def test_stride_none_valid_padding(self, stride, kernel, dilation):
        """Test get_padding_and_stride, when stride is None and padding mode is VALID"""
        with pytest.raises(ValueError, match="Stride must be specified for VALID padding mode."):
            get_padding_and_stride(kernel, dilation, PaddingMode.VALID, num_dims=2, stride=stride)

    @pytest.mark.parametrize(
        "padding_mode,kernel,dilation,stride_arg,expected_stride,expected_padding", [
            # VALID mode cases (stride preserved, zero padding)
            (PaddingMode.VALID, [3, 5], [1, 1], 1, 1, (0, 0)),
            # SAME mode cases (stride forced to 1)
            #odd kernels, odd dilation result in integer padding
            (PaddingMode.SAME, 3, 1, 2, 1, (1, 1)),  # kernel=3, dilation=1 -> padding=1
            # even kernels, even dilation result in integer padding
            (PaddingMode.SAME, 4, 2, 3, 1, (3, 3)),  # kernel=4, dilation=2 -> padding=3
            # odd kernels, even dilation result in integer padding
            (PaddingMode.SAME, 5, 2, 4, 1, (4, 4)),  # kernel=5, dilation=2 -> padding=4
            # DOWNSAMPLING_FACTOR_2 mode cases (stride forced to 2)
            # even kernels, odd dilations result in integer padding
            (PaddingMode.DOWNSAMPLING_FACTOR_2, 2, 1, 5, 2, (0, 0)),   # kernel=2, dilation=1 -> padding=0
        ])
    def test_padding_modes_unified(self, padding_mode, kernel, dilation, stride_arg, expected_stride, expected_padding):
        """
        Unified test covering VALID, SAME, and DOWNSAMPLING_FACTOR_2 modes for the default value of num_dims=2.
        """
        # For VALID mode stride is mandatory; for others we still pass it intentionally to ensure overriding logic.
        params = get_padding_and_stride(kernel=kernel, dilation=dilation, padding_mode=padding_mode, stride=stride_arg if padding_mode == PaddingMode.VALID else stride_arg)
        assert params.stride == expected_stride
        assert params.padding == expected_padding