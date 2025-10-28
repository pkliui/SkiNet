from dataclasses import dataclass
from typing import Iterable, Tuple, Union, Optional


@dataclass
class UpsamplingParams:
    """
    Holds the parameters for the upsampling operation.
    """
    upsampling_kernel: Union[int, Iterable[int]]
    upsampling_stride: Union[int, Iterable[int]]
    upsampling_padding: Optional[Union[Tuple[int, ...], str]] = None


@dataclass
class EncoderSpec:
    """Simple container describing the encoder convolution that we want to invert.

    :param kernel: Kernel size used in the encoder Conv2d (int or iterable)
    :param stride: Stride used in the encoder Conv2d (int or iterable)
    :param padding: (Optional) Numeric padding used in the encoder Conv2d (int/tuple) — kept for compatibility but not used when a PaddingMode is available
    """
    kernel: Union[int, Iterable[int]] = None
    stride: Union[int, Iterable[int]] = None
    padding: Optional[Union[int, Iterable[int], str]] = None


def validate_encoder_spec(encoder_spec: Optional[EncoderSpec]) -> None:
    """
    Validates that all required fields in EncoderSpec are present and not None.
    Raises ValueError if any required field is missing.
    """
    required_fields = ['stride', 'kernel', 'padding']
    if encoder_spec is None:
        raise ValueError("encoder_spec must be provided.")
    for field in required_fields:
        if getattr(encoder_spec, field) is None:
            raise ValueError(f"encoder_spec.{field} must be specified. Got None.")


def compute_convtranspose2d_params(encoder_spec: Optional[EncoderSpec] = None) -> UpsamplingParams:
    """
    Compute ConvTranspose2d parameters for the decoder

    :param encoder_spec: EncoderSpec describing the corresponding encoder convolution to be inverted.
    """
    validate_encoder_spec(encoder_spec)

    upsampling_kernel = encoder_spec.kernel
    upsampling_stride = encoder_spec.stride
    # padding is set to the same as encoder padding to ensure we can recover the original input size
    # see documentation note for https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
    upsampling_padding = encoder_spec.padding

    return UpsamplingParams(upsampling_kernel=upsampling_kernel,
                            upsampling_stride=upsampling_stride,
                            upsampling_padding=upsampling_padding)