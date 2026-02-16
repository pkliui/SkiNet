from typing import Callable, Optional

import pytest
from torch import all, isfinite, nn, randn, sqrt

from SkiNet.ML.model.blocks.conv2d_layer import Conv2dLayer
from SkiNet.ML.utils.sampling.encoder_sampling import get_encoder_params_2d


@pytest.mark.parametrize("batch_size, in_channels, out_channels, kernel, stride, dilation, apply_bias, apply_batchnorm, activation, input_size, expected_size",
                         [
                             (2, 3, 6, 3, 1, 1, False, True, nn.ReLU, 10, 10),  # stride=1,
                             (2, 3, 6, 3, 1, 2, False, True, nn.ReLU, 10, 10),
                             (2, 6, 12, 3, 2, 1, False, True, nn.ReLU, 20, 10),  # stride=2,
                             (2, 6, 12, 3, 2, 2, False, True, nn.ReLU, 20, 10),
                             (2, 6, 12, 3, 2, 1, True, False, None, 20, 10),  # no BN, no activation
                         ])
def test_basic_layer_forward_and_backward_pass(batch_size: int,
                                               in_channels: int,
                                               out_channels: int,
                                               kernel: int,
                                               stride: int,
                                               dilation: int,
                                               apply_bias: bool,
                                               apply_batchnorm: bool,
                                               activation: type[nn.Module],
                                               input_size: int,
                                               expected_size: int) -> None:
    """
    Test forward and backward passes in Conv2dLayer layer
    """
    input = randn(batch_size, in_channels, input_size, input_size).float()
    ground_truth = randn(batch_size, out_channels, expected_size, expected_size).float()

    params = get_encoder_params_2d(kernel=kernel, stride=stride, dilation=dilation)
    layer = Conv2dLayer(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel=params.kernel,
                        stride=params.stride,
                        dilation=params.dilation,
                        padding=params.padding,
                        apply_bias=apply_bias,
                        apply_batchnorm=apply_batchnorm,
                        activation=activation)
    output = layer(input)
    loss = sqrt(nn.MSELoss()(output, ground_truth) + 1e-8)

    # Check backward pass
    loss.backward()

    # Check output size
    assert output.shape == (batch_size, out_channels, expected_size, expected_size)

    # Loss is a positive finite value
    assert isinstance(loss.item(), float)
    assert loss.item() > 0
    assert isfinite(loss).item() is True

    # Positive-valued output after ReLU
    if activation is not None:
        assert all(output >= 0.0)
        assert isfinite(output).all()


@pytest.mark.parametrize(
    "activation",
    [
        nn.ReLU,
        lambda: nn.ReLU(inplace=True),
        None
    ],
)
def test_conv2d_layer_accepts_relu_and_relu_inplace(activation: Optional[Callable[[], nn.Module]]) -> None:
    batch_size = 2
    in_channels = 3
    out_channels = 6
    kernel = 3
    stride = 1
    dilation = 1
    input_size = 10
    expected_size = 10

    input = randn(batch_size, in_channels, input_size, input_size).float()
    ground_truth = randn(batch_size, out_channels, expected_size, expected_size).float()

    params = get_encoder_params_2d(kernel=kernel, stride=stride, dilation=dilation)

    layer = Conv2dLayer(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel=params.kernel,
                        stride=params.stride,
                        dilation=params.dilation,
                        padding=params.padding,
                        apply_bias=False,
                        apply_batchnorm=True,
                        activation=activation)

    output = layer(input)
    loss = sqrt(nn.MSELoss()(output, ground_truth) + 1e-8)

    # Check backward pass
    loss.backward()

    # Check output size
    assert output.shape == (batch_size, out_channels, expected_size, expected_size)

    # Loss is a positive finite value
    assert isinstance(loss.item(), float)
    assert loss.item() > 0
    assert isfinite(loss).item() is True

    # Positive-valued output after ReLU
    if activation is not None:
        assert all(output >= 0.0)
        assert isfinite(output).all()
