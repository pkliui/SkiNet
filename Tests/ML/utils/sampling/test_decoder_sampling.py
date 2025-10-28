import pytest
from SkiNet.ML.utils.sampling.decoder_sampling import EncoderSpec, compute_convtranspose2d_params, UpsamplingParams, validate_encoder_spec


######################################## Tests for UpsamplingParams ########################################

class TestUpsamplingParams:
    """
    Unit tests for UpsamplingParams dataclass, covering instantiation with various argument types.
    """
    def test_tuple_inputs(self):
        """
        Test instantiation with tuple arguments for all fields.
        """
        params = UpsamplingParams(
            upsampling_kernel=(3, 3),
            upsampling_stride=(2, 2),
            upsampling_padding=(1, 1)
        )
        assert params.upsampling_kernel == (3, 3)
        assert params.upsampling_stride == (2, 2)
        assert params.upsampling_padding == (1, 1)

    def test_str_tuple_int_inputs(self):
        """
        Test instantiation with int kernel and stride and string padding.
        """
        params = UpsamplingParams(
            upsampling_kernel=(4,4),
            upsampling_stride=[2,2],
            upsampling_padding="same")
        assert params.upsampling_kernel == (4,4)
        assert params.upsampling_stride == [2,2]
        assert params.upsampling_padding == "same"


######################################## Tests for EncoderSpec ########################################

class TestEncoderSpec:
    """
    Unit tests for the EncoderSpec dataclass, covering instantiation and field access.
    """
    def test_tuple_inputs(self):
        """
        Test instantiation with tuple arguments for all fields.
        """
        spec = EncoderSpec(
            kernel=(3, 3),
            stride=(2, 2),
            padding=(1, 1)
        )
        assert spec.kernel == (3, 3)
        assert spec.stride == (2, 2)
        assert spec.padding == (1, 1)

    def test_str_and_int_inputs(self):
        """
        Test instantiation with int kernel and stride and string padding.
        """
        spec = EncoderSpec(
            kernel=(4,4),
            stride=(2,2),
            padding="same"
        )
        assert spec.kernel == (4,4)
        assert spec.stride == (2,2)
        assert spec.padding == "same"

    def test_optional_fields(self):
        """
        Test instantiation with only required fields and default values for optional fields.
        """
        spec = EncoderSpec(
            kernel=(5, 5),
            stride=(2, 2)
        )
        assert spec.kernel == (5, 5)
        assert spec.stride == (2, 2)
        assert spec.padding is None

def test_validate_encoder_spec_raises_on_missing_fields():
    """
    Test missing fields in EncoderSpec raise ValueError.
    """
    # Missing 'stride'
    spec_missing_stride = EncoderSpec(kernel=(5, 5), stride=None, padding=(1, 1))
    with pytest.raises(ValueError, match="encoder_spec.stride must be specified. Got None."):
        validate_encoder_spec(spec_missing_stride)

    # Missing 'kernel'
    spec_missing_kernel = EncoderSpec(kernel=None, stride=(2, 2), padding=(1, 1))
    with pytest.raises(ValueError, match="encoder_spec.kernel must be specified. Got None."):
        validate_encoder_spec(spec_missing_kernel)

    # Missing 'padding'
    spec_missing_padding = EncoderSpec(kernel=(5, 5), stride=(2, 2), padding=None)
    with pytest.raises(ValueError, match="encoder_spec.padding must be specified. Got None."):
        validate_encoder_spec(spec_missing_padding)

    # encoder_spec is None
    with pytest.raises(ValueError, match="encoder_spec must be provided."):
        validate_encoder_spec(None)

######################################## Tests for compute_convtranspose2d_params ########################################

class TestComputeConvTranspose2DParams:
    """
    Unit tests for compute_convtranspose2d_params, covering both upsampling modes and error cases.
    """
    @pytest.mark.parametrize(
        "kernel, stride, padding, expected_upsampling_kernel, expected_upsampling_stride, expected_upsampling_padding",
        [
            ((4, 4), (2, 2), (1, 1), (4, 4), (2, 2), (1, 1)),
            ((6, 6), (2, 2), (1, 1), (6, 6), (2, 2), (1, 1)),
        ]
    )
    def test_compute_convtranspose2d_params(self, kernel, stride, padding, expected_upsampling_kernel, expected_upsampling_stride, expected_upsampling_padding):
        """
        Test compute_convtranspose2d_params returns correct kernel, stride, and numeric padding; and correct UpsamplingParams type.
        """
        encoder_spec = EncoderSpec(
            kernel=kernel,
            stride=stride,
            padding=padding,
        )
        params = compute_convtranspose2d_params(encoder_spec=encoder_spec)
        assert params.upsampling_kernel == expected_upsampling_kernel
        assert params.upsampling_stride == expected_upsampling_stride
        assert params.upsampling_padding == expected_upsampling_padding
        assert isinstance(params, UpsamplingParams)
