import numpy as np
import torch
import pytest
from torch.nn import MSELoss

from SkiNet.ML.model.tr_conv2d_layer import TrConv2dLayer
from SkiNet.ML.utils.sampling.encoder_sampling import PaddingMode, get_padding
from SkiNet.ML.utils.sampling.decoder_sampling import EncoderSpec


class TestTrConv2dLayer:

    def test_valid_construction(self):
        """
        Test that TrConv2dLayer is composed of a conv2d layer, batchnorm2d, and an activation function
        """
        spec = EncoderSpec(kernel=[4, 4], stride=[2, 2], padding=[1, 1])
        activation = torch.nn.ReLU
        layer = TrConv2dLayer(
            in_channels=1,
            out_channels=1,
            encoder_spec=spec,
            activation=activation
        )

        assert any([isinstance(ll, torch.nn.ConvTranspose2d) for ll in layer.children()])
        assert any([isinstance(ll, torch.nn.BatchNorm2d) for ll in layer.children()])
        assert any([isinstance(ll, activation) for ll in layer.children()])
        if hasattr(layer.activation, "inplace"):
            assert layer.activation.inplace is True

    def test_invalid_encoder_spec_raises(self):
        """
        Test that providing an invalid encoder_spec raises a ValueError
        """
        with pytest.raises(ValueError):
            TrConv2dLayer(
                in_channels=4,
                out_channels=8,
                encoder_spec="not_a_valid_spec",
                activation=torch.nn.ReLU
            )


@pytest.mark.parametrize("batch_size, in_channels, out_channels, input_shape, kernel, stride, dilation, padding_mode", [
    (2, 3, 6, (512, 512), (4, 4), (2, 2), (1,1), PaddingMode.DOWNSAMPLING_FACTOR_2),
    (2, 3, 6, (512, 512), (6, 6), (2, 2), (1,1), PaddingMode.DOWNSAMPLING_FACTOR_2),
    (2, 3, 6, (512, 512), (8, 8), (2, 2), (1,1), PaddingMode.DOWNSAMPLING_FACTOR_2)
])
def test_basic_trconv2d_forward_and_backward_pass(batch_size, in_channels, out_channels, input_shape, kernel, stride, dilation, padding_mode):
    """
    Test forward and backward passes in TrConv2dLayer with upsampling.
    Assumes that the upsampling will upsample the spatial dimensions of the input as per given stride.
    """
    padding = get_padding(kernel=kernel,
                        dilation=dilation,
                        padding_mode=padding_mode,
                        num_dims=2,
                        stride=stride)

    encoder_spec = EncoderSpec(kernel=kernel,
        stride=stride,
        padding=padding)

    layer = TrConv2dLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        encoder_spec=encoder_spec,
        activation=torch.nn.ReLU
    )

    # Forward pass
    input = torch.rand(batch_size, in_channels, *input_shape).float()
    output = layer(input)

    # Output should be upsampled
    assert output.shape[0] == batch_size
    assert output.shape[1] == out_channels
    assert output.shape[2] > input_shape[0] and output.shape[3] > input_shape[1]
    assert output.shape == input.shape[:1] + (out_channels, input_shape[0] * stride[0], input_shape[1] * stride[1])

    # Loss and backward
    ground_truth = torch.rand(batch_size, out_channels, input_shape[0] * stride[0], input_shape[1] * stride[1]).float()
    loss_function = MSELoss()
    loss = loss_function(output, ground_truth)

    # Loss is a positive finite value
    assert isinstance(loss.item(), float)
    assert loss.item() > 0
    assert not np.isnan(loss.detach().numpy()).any()
    assert not np.isinf(loss.detach().numpy()).any()

    # Positive-valued output after ReLU
    assert np.all(output.detach().numpy() >= 0.0)
    assert not np.isnan(output.detach().numpy()).any()
    assert not np.isinf(output.detach().numpy()).any()
