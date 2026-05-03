import pytest
from pydantic import ValidationError

from SkiNet.ML.configs.model_configs.unet2d_config import UNet2DModelConfig
from SkiNet.ML.utils.typing_utils import IntOrTuple2d


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        # Valid config, all defaults
        ({}, {"in_channels": 3, "out_channels_layer1": 16, "number_of_layers": 5, "num_output_classes": 1,
              "kernel": 3, "stride": 2, "dilation": 1, "model_name": "UNet2D", "validate_forward": True,
              "debug_forward": False, "encoder_residual_mode": "he2", "merge_residual_mode": "he2", "kind": "unet2d"}),
        # Valid config, custom values
        ({"in_channels": 3, "out_channels_layer1": 8, "number_of_layers": 4, "num_output_classes": 2,
          "kernel": (3, 3), "stride": (2, 2), "dilation": (1, 1), "model_name": "UNet2D", "validate_forward": True,
          "debug_forward": True},
         {"in_channels": 3, "out_channels_layer1": 8, "number_of_layers": 4, "num_output_classes": 2,
          "kernel": (3, 3), "stride": (2, 2), "dilation": (1, 1), "model_name": "UNet2D", "validate_forward": True,
          "debug_forward": True, "encoder_residual_mode": "he2", "merge_residual_mode": "he2", "kind": "unet2d"}),
        # Valid config, non-default residual modes
        ({"encoder_residual_mode": "local_refinement", "merge_residual_mode": "he1"},
         {"encoder_residual_mode": "local_refinement", "merge_residual_mode": "he1"}),
        ({"encoder_residual_mode": "local_refinement", "merge_residual_mode": "local_refinement"},
         {"encoder_residual_mode": "local_refinement", "merge_residual_mode": "local_refinement"}),
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
        # Invalid: in_channels must be 3 for RGB images
        ({"in_channels": 1}, "in_channels"),
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
        # Invalid: bad residual mode strings
        ({"encoder_residual_mode": "bad_mode"}, "encoder_residual_mode"),
        ({"merge_residual_mode": "bad_mode"}, "merge_residual_mode"),
    ]
)
def test_unet2dmodelconfig_invalid(kwargs: dict, error_field: str) -> None:
    """
    Test UNet2DModelConfig with invalid parameters.
    """
    with pytest.raises(ValidationError) as e:
        UNet2DModelConfig(**kwargs)
    assert error_field in str(e.value)


@pytest.mark.parametrize(
    "kwargs, expected_down_layers, expected_required_multiple",
    [
        ({}, 4, 16),  # number_of_layers=5 => down=4, stride=2 => 2**4=16
        ({"number_of_layers": 4}, 3, 8),  # down=3 => 2**3=8
        ({"number_of_layers": 6, "stride": 2}, 5, 32),  # down=5 => 2**5=32
        ({"number_of_layers": 5, "stride": (2, 2)}, 4, (16, 16)),
        ({"number_of_layers": 5, "stride": (2, 4)}, 4, (16, 256)),
    ],
)
def test_unet2dmodelconfig_derived_properties(kwargs: dict,
                                              expected_down_layers: int,
                                              expected_required_multiple: IntOrTuple2d) -> None:
    """
    Test derived properties of UNet2DModelConfig.
    """
    cfg = UNet2DModelConfig(**kwargs)
    assert cfg.number_of_downsampling_layers == expected_down_layers
    assert cfg.required_input_multiple == expected_required_multiple
