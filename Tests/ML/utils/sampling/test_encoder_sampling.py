import pytest
from typing import Iterable, Tuple, Union

from SkiNet.ML.utils.sampling.encoder_sampling import PaddingMode, get_padding_stride_2, get_padding_value, get_same_padding_stride_1, handle_mixed_inputs


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

    def test_mixed_inputs_different_lengths(self):
        """Test mixed inputs with different lengths - should raise ValueError with strict validation"""
        with pytest.raises(ValueError, match="All iterable inputs must have the same size"):
            handle_mixed_inputs([3, 5], [1, 2, 3, 4])

    def test_single_element_iterables(self):
        """Test with single-element iterables"""
        kernel, dilation = handle_mixed_inputs([3], (1,))
        
        assert kernel == [3]
        assert dilation == [1]

    def test_preserves_iterable_types(self):
        """Test that iterable types are preserved (converted to lists)"""
        kernel, dilation = handle_mixed_inputs((3, 5), [1, 2])
        
        assert kernel == [3, 5]  # tuple converted to list
        assert dilation == [1, 2]  # list preserved as list

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

    def test_negative_values(self):
        """Test with negative values - should raise ValueError"""
        with pytest.raises(ValueError, match="kernel must be positive, got -1"):
            handle_mixed_inputs(-1, [2, 3])

    def test_large_num_dims(self):
        """Test with large num_dims value"""
        kernel, dilation = handle_mixed_inputs(3, 1, num_dims=10)
        
        assert kernel == [3] * 10
        assert dilation == [1] * 10
        assert len(kernel) == 10
        assert len(dilation) == 10

    def test_num_dims_ignored_when_iterables_present(self):
        """Test that num_dims is ignored when iterables are present"""
        kernel, dilation = handle_mixed_inputs([3, 5], 1, num_dims=10)
        
        # Should use max length (2) from the iterable, not num_dims (10)
        assert kernel == [3, 5]
        assert dilation == [1, 1]
        assert len(kernel) == 2
        assert len(dilation) == 2

    def test_return_type_consistency(self):
        """Test that return types are consistent (all lists)"""
        kernel, dilation = handle_mixed_inputs((3, 5), [1, 2])
        
        assert isinstance(kernel, list)
        assert isinstance(dilation, list)

    def test_tuple_inputs(self):
        """Test with tuple inputs - should raise ValueError for different sizes"""
        with pytest.raises(ValueError, match="All iterable inputs must have the same size"):
            handle_mixed_inputs((3, 5, 7), (1, 2))

    def test_list_inputs(self):
        """Test with list inputs - should raise ValueError for different sizes"""
        with pytest.raises(ValueError, match="All iterable inputs must have the same size"):
            handle_mixed_inputs([3, 5], [1, 2, 3, 4])


class TestGetPaddingValue:
    """Test cases for get_padding_value function"""

    def test_get_padding_value_wrong_sampling_mode(self):
        """Test get_padding_value, when samping mode is not in PaddingMode"""
        test_cases = [
            "invalid_mode",
            "DOWNSAMPLING_FACTOR_22",
        ]
        
        for sampling_mode  in test_cases:
            with pytest.raises(ValueError, match=f"Invalid sampling mode:"):
                _ = get_padding_value(1,3,1, sampling_mode)

    
    def test_get_padding_value_stride_not_positive(self):
        """Test get_padding_value, when stride is not 1 or 2 - should raise ValueError"""
        test_cases = [
            (0, 3, 1),
            (-1, 3, 1),
            (3, 3, 1),
            (4, 3, 1),
        ]
        
        for stride, kernel, dilation  in test_cases:
            # Function now validates stride values
            with pytest.raises(ValueError, match=f"Stride must be 1 or 2, got {stride}"):
                get_padding_value(stride, kernel, dilation, PaddingMode.VALID)
    
    def test_get_padding_value_same_mode_stride_not_1(self):
        """Test get_padding_value, when stride is not 1 and SAME mode"""
        test_cases = [
            (2, 3, 1), #stride=2 - should fail at mode level
            (3, 3, 1), #stride=3 - should fail at stride validation level
        ]
        
        for stride, kernel, dilation  in test_cases:
            if stride == 2:
                # stride=2 is valid but not supported for SAME mode
                with pytest.raises(ValueError, match=f"Unsupported stride {stride} for SAME sampling. Only stride=1 is supported."):
                    _ = get_padding_value(stride, kernel, dilation, PaddingMode.SAME)
            else:
                # stride=3 is invalid at stride validation level
                with pytest.raises(ValueError, match=f"Stride must be 1 or 2, got {stride}"):
                    _ = get_padding_value(stride, kernel, dilation, PaddingMode.SAME)

    def test_get_padding_value_same_mode_stride_not_2(self):
        """Test get_padding_value, when stride is not 2 and DOWNSAMPLING_FACTOR_2 mode"""
        test_cases = [
            (1, 3, 1),#stride=1 - should fail at mode level
            (3, 3, 1),#stride=3 - should fail at stride validation level
        ]
        
        for stride, kernel, dilation  in test_cases:
            if stride == 1:
                # stride=1 is valid but not supported for DOWNSAMPLING_FACTOR_2 mode
                with pytest.raises(ValueError, match=f"Unsupported stride {stride} for DOWNSAMPLING_FACTOR_2 sampling. Only stride=2 is supported."):
                    _ = get_padding_value(stride, kernel, dilation, PaddingMode.DOWNSAMPLING_FACTOR_2)
            else:
                # stride=3 is invalid at stride validation level
                with pytest.raises(ValueError, match=f"Stride must be 1 or 2, got {stride}"):
                    _ = get_padding_value(stride, kernel, dilation, PaddingMode.DOWNSAMPLING_FACTOR_2)


    def test_get_padding_value_valid_sampling_mode(self):
        """Test get_padding_value with VALID sampling mode returns 0 padding"""
        test_cases = [
            (1, 3, 1),  # stride=1, kernel=3, dilation=1
            (1, 5, 1),  # stride=1, kernel=5, dilation=1
            (1, 3, 2),  # stride=1, kernel=3, dilation=2
            (1, 5, 2),  # stride=1, kernel=5, dilation=2
            (2, 3, 1),  # stride=2, kernel=3, dilation=1
            (2, 5, 1),  # stride=2, kernel=5, dilation=1
        ]
        
        for stride, kernel, dilation in test_cases:
            padding = get_padding_value(stride, kernel, dilation, PaddingMode.VALID)
            # Function returns tuple due to handle_mixed_inputs processing
            assert padding == (0, 0)    

    
    def test_get_padding_value_same_stride_1_equals_get_same_padding_stride_1(self):
        """Test that get_padding_value with stride=1 and SAME mode returns same result as get_same_padding_stride_1"""
        test_cases = [
            (1, 3, 1),   # stride=1, kernel=3, dilation=1
            (1, 5, 1),   # stride=1, kernel=5, dilation=1
            (1, 7, 1),   # stride=1, kernel=7, dilation=1
        ]
        
        for stride, kernel, dilation in test_cases:
            padding_from_get_padding_value = get_padding_value(stride, kernel, dilation, PaddingMode.SAME)
            # Individual functions now handle input conversion internally
            padding_from_get_same_padding_stride_1 = get_same_padding_stride_1(kernel, dilation)
            
            assert padding_from_get_padding_value == padding_from_get_same_padding_stride_1, \
                f"stride={stride}, kernel={kernel}, dilation={dilation}: " \
                f"get_padding_value returned {padding_from_get_padding_value}, " \
                f"get_same_padding_stride_1 returned {padding_from_get_same_padding_stride_1}"


    def test_get_padding_value_same_stride_2_equals_get_padding_stride_2(self):
        """Test that get_padding_value with stride=2 and DOWNSAMPLING_FACTOR_2 mode returns same result as get_padding_stride_2"""
        test_cases = [
            (2, 2, 1, 0),   # kernel=2, dilation=1 -> padding=0
            (2, 4, 1, 1),   # kernel=4, dilation=1 -> padding=1
            (2, 6, 1, 2),   # kernel=6, dilation=1 -> padding=2
            (2, 2, 3, 1),   # kernel=2, dilation=3 -> padding=2
            (2, 4, 3, 4),   # kernel=4, dilation=3 -> padding=4
            (2, 6, 3, 7),   # kernel=6, dilation=3 -> padding=7
        ]
        
        for stride, kernel, dilation,_ in test_cases:
            # Test that both functions return the same result
            padding_from_get_padding_value = get_padding_value(stride, kernel, dilation, PaddingMode.DOWNSAMPLING_FACTOR_2)
            # Individual functions now handle input conversion internally
            padding_from_get_padding_stride_2 = get_padding_stride_2(kernel, dilation)
            
            assert padding_from_get_padding_value == padding_from_get_padding_stride_2, \
                f"stride={stride}, kernel={kernel}, dilation={dilation}: " \
                f"get_padding_value returned {padding_from_get_padding_value}, " \
                f"get_padding_stride_2 returned {padding_from_get_padding_stride_2}"

######################################## Tests for SAME sampling mode with stride=1 ########################################

    def test_get_same_padding_stride_1_dilation_kernel_negative_error(self):
        """
        Test get_same_padding_stride_1 with stride=1, when dilation and kernel are negative-valued

        :raise ValueError: for s=1, k<0, d<0; n=1,2,3,...
        """
        # Test cases: (stride, kernel, dilation)
        test_cases = [
            (1, -2, -1),   # kernel=-2, dilation=-1
            (1, 2, -1),   # kernel=2, dilation=-1
            (1, -2, 1),   # kernel=-2, dilation=1
        ]
        
        for _, kernel, dilation  in test_cases:
            # Individual functions now call handle_mixed_inputs which validates inputs
            with pytest.raises(ValueError, match="must be positive"):
                _ = get_same_padding_stride_1(kernel, dilation)

    def test_get_same_padding_stride_1_dilation_1_kernel_even_error(self):
        """
        Test get_same_padding_stride_1, where for stride=1, dilation=1, kernels are even and hence result in fractional padding

        :raise ValueError: for s=1, k=2n, d=1; n=1,2,3,...
        """
        # Test cases: (stride, kernel, dilation, expected_padding)
        test_cases = [
            (1, 2, 1, 1/2),   # stride=1, kernel=2, dilation=1 -> padding=1/2
            (1, 4, 1, 3/2),   # stride=1, kernel=4, dilation=1 -> padding=3/2
            (1, 6, 1, 5/2),   # stride=1, kernel=6, dilation=1 -> padding=5/2
        ]
        
        for _, kernel, dilation, padding  in test_cases:
            # Individual functions now handle input conversion internally
            with pytest.raises(ValueError, match="not all items in padding"):
                _ = get_same_padding_stride_1(kernel, dilation)

    def test_get_same_padding_stride_1_dilation_1_kernel_odd(self):
        """
        Test get_same_padding_stride_1, where stride=1 and dilation=1 and odd kernels result in integer padding
        """
        # Test cases: (stride, kernel, dilation, expected_padding)
        test_cases = [
            (1, 3, 1, 1),   # stride=1, kernel=3, dilation=1 -> padding=1
            (1, 5, 1, 2),   # stride=1, kernel=5, dilation=1 -> padding=2
            (1, 7, 1, 3),   # stride=1, kernel=7, dilation=1 -> padding=3
        ]
        
        for _, kernel, dilation, expected_padding in test_cases:
            # Individual functions now handle input conversion internally
            padding = get_same_padding_stride_1(kernel, dilation)
            # Function returns tuple, so we need to check the first element
            assert padding[0] == expected_padding


    def test_get_same_padding_stride_1_dilation_2_kernel_odd(self):
        """
        Test get_same_padding_stride_1, where stride=1 and dilation=2 and odd kernels result in integer padding
        """
        # Test cases: (stride, kernel, dilation, expected_padding)
        test_cases = [
            (1, 3, 2, 2),   # stride=1, ,kernel=3, dilation=2 -> padding=2
            (1, 5, 2, 4),   # stride=1, kernel=5, dilation=2 -> padding=4
            (1, 7, 2, 6),   # stride=1, kernel=7, dilation=2 -> padding=6
        ]
        
        for _, kernel, dilation, expected_padding in test_cases:
            # Individual functions now handle input conversion internally
            padding = get_same_padding_stride_1(kernel, dilation)
            # Function returns tuple, so we need to check the first element
            assert padding[0] == expected_padding


    def test_get_same_padding_stride_1_dilation_2_kernel_even(self):
        """Test get_same_padding_stride_1, where  stride=1 and dilation=2, and even kernels, result in integer padding
        """
        # Test cases: (stride, kernel, dilation, expected_padding)
        test_cases = [
            (1, 2, 2, 1),   # stride=1, kernel=2, dilation=2 -> padding=1
            (1, 4, 2, 3),   #stride=1,  kernel=4, dilation=2 -> padding=3
            (1, 6, 2, 5),   # stride=1, kernel=6, dilation=2 -> padding=5
        ]
        
        for _, kernel, dilation, expected_padding in test_cases:
            # Individual functions now handle input conversion internally
            padding = get_same_padding_stride_1(kernel, dilation)
            # Function returns tuple, so we need to check the first element
            assert padding[0] == expected_padding


######################################## Tests for DOWNSAMPLING_FACTOR_2 sampling mode with stride=2 ########################################

    
    def test_get_padding_stride_2_dilation_kernel_negative_error(self):
        """Test get_padding_stride_2, where stride=2, dilation and kernel negative values
        :raise ValueError: for s=2, k<0, d<0; n=1,2,3,...
        """
        # Test cases: (stride, kernel, dilation) - using negative values
        test_cases = [
            (2, -2, -1),   # kernel=-2, dilation=-1
            (2, 2, -1),   # kernel=2, dilation=-1
            (2, -2, 1),   # kernel=-2, dilation=1
        ]
        
        for _, kernel, dilation  in test_cases:
            # Individual functions now call handle_mixed_inputs which validates inputs
            with pytest.raises(ValueError, match="must be positive"):
                _ = get_padding_stride_2(kernel, dilation)

    def test_get_padding_stride_2_kernel_odd_error(self):
        """
        Test get_padding_stride_2, where for stride=2, dilation odd, kernels are odd and hence result in fractional padding
        
        :raise ValueError: for s=2, k=2n+1, d=2n+1; n=1,2,3,...
        """
        # Test cases: (stride, kernel, dilation, expected_padding)
        test_cases = [
            (2, 3, 1, 1/2),   # kernel=3, dilation=1 -> padding=1/2
            (2, 5, 1, 3/2),   # kernel=5, dilation=1 -> padding=3/2
            (2, 7, 1, 5/2),   # kernel=7, dilation=1 -> padding=5/2
            (2, 3, 3, 5/2),   # kernel=3, dilation=3 -> padding=5/2
            (2, 5, 3, 11/2),   # kernel=5, dilation=3 -> padding=11/2
            (2, 7, 3, 17/2),   # kernel=7, dilation=3 -> padding=17/2
        ]
        
        for _, kernel, dilation, padding  in test_cases:
            # Individual functions now handle input conversion internally
            with pytest.raises(ValueError, match="not all items in padding"):
                _ = get_padding_stride_2(kernel, dilation)


    def test_get_padding_stride_2_dilation_even_error(self):
        """
        Test get_padding_stride_2, where for stride=2, even kernels,  dilations are even and hence result in fractional padding
        
        :raise ValueError: for s=2, k=2n, d=2n; n=1,2,3,...
        """
        # Test cases: (stride, kernel, dilation, expected_padding)
        test_cases = [
            (2, 2, 2, 1/2),  
            (2, 4, 2, 5/2), 
        ]
        
        for _, kernel, dilation, padding  in test_cases:
            # Individual functions now handle input conversion internally
            with pytest.raises(ValueError, match="not all items in padding"):
                _ = get_padding_stride_2(kernel, dilation)

    def test_get_padding_stride_2_kernel_odd_dilation_even_error(self):
        """
        Test get_padding_stride_2, where for stride=2, odd kernels, even dilations result in fractional padding
        
        :raise ValueError: for s=2, k=2n+1, d=2m; n,m=1,2,3,...
        """
        # Test cases: (stride, kernel, dilation, expected_padding)
        test_cases = [
            (2, 3, 2, 3/2),   # kernel=3, dilation=2 -> padding = ((3-1)*2-1)/2 = 3/2
            (2, 5, 2, 7/2),   # kernel=5, dilation=2 -> padding = ((5-1)*2-1)/2 = 7/2
            (2, 3, 4, 7/2),   # kernel=3, dilation=4 -> padding = ((3-1)*4-1)/2 = 7/2
        ]
        
        for _, kernel, dilation, padding  in test_cases:
            # Individual functions now handle input conversion internally
            with pytest.raises(ValueError, match="not all items in padding"):
                _ = get_padding_stride_2(kernel, dilation)
    
    def test_get_padding_stride_2_kernel_even_dilation_odd(self):
        """
        Test get_padding_stride_2, where stride=2, dilation={1,3}, kernels={2,4,6} result in integer padding
        """
        # Test cases: (stride, kernel, dilation, expected_padding)
        test_cases = [
            (2, 2, 1, 0),   # kernel=2, dilation=1 -> padding=0
            (2, 4, 1, 1),   # kernel=4, dilation=1 -> padding=1
            (2, 6, 1, 2),   # kernel=6, dilation=1 -> padding=2
            (2, 2, 3, 1),   # kernel=2, dilation=3 -> padding=2
            (2, 4, 3, 4),   # kernel=4, dilation=3 -> padding=4
            (2, 6, 3, 7),   # kernel=6, dilation=3 -> padding=7
        ]
        
        for _, kernel, dilation, expected_padding in test_cases:
            # Individual functions now handle input conversion internally
            padding = get_padding_stride_2(kernel, dilation)
            # Function returns tuple, so we need to check the first element
            assert padding[0] == expected_padding
    