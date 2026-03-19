import pytest
from pydantic import ValidationError

from SkiNet.ML.configs.model_configs.unet2d_config import UNet2DModelConfig


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        # Valid config, all defaults
        ({}, {"in_channels": 1, "out_channels_layer1": 16, "number_of_layers": 5, "num_output_classes": 1,
              "kernel": 3, "stride": 2, "dilation": 1, "model_name": "UNet2D", "validate_forward": False, "kind": "unet2d"}),
        # Valid config, custom values
        ({"in_channels": 3, "out_channels_layer1": 8, "number_of_layers": 4, "num_output_classes": 2,
          "kernel": (3, 3), "stride": (2, 2), "dilation": (1, 1), "model_name": "UNet2D", "validate_forward": True},
         {"in_channels": 3, "out_channels_layer1": 8, "number_of_layers": 4, "num_output_classes": 2,
          "kernel": (3, 3), "stride": (2, 2), "dilation": (1, 1), "model_name": "UNet2D", "validate_forward": True, "kind": "unet2d"}),
    ]
)
def test_unet2dmodelconfig_valid(kwargs: dict, expected: dict) -> None:
    """
    Test UNet2DModelConfig with valid parameters.
    """
    cfg = UNet2DModelConfig(**kwargs)
    for k, v in expected.items():
        assert getattr(cfg, k) == v

@pytest.mark.parametrize(
    "kwargs, error_field",
    [
        # Invalid: in_channels < 1
        ({"in_channels": 0}, "in_channels"),
        # Invalid: out_channels_layer1 < 1
        ({"out_channels_layer1": 0}, "out_channels_layer1"),
        # Invalid: number_of_layers < 2
        ({"number_of_layers": 1}, "number_of_layers"),
        # Invalid: num_output_classes < 1
        ({"num_output_classes": 0}, "num_output_classes"),
        # Invalid: kernel even (should fail in validate_conv_inputs)
        ({"kernel": 2}, "kernel"),
        # Invalid: stride tuple wrong length
        ({"stride": (2, 2, 2)}, "stride"),
    ]
)
def test_unet2dmodelconfig_invalid(kwargs: dict, error_field: str) -> None:
    """
    Test UNet2DModelConfig with invalid parameters.
    """
    with pytest.raises(ValidationError) as e:
        UNet2DModelConfig(**kwargs)
    assert error_field in str(e.value)
