import torch
from typing import Optional, Callable
from SkiNet.ML.utils.sampling.decoder_sampling import compute_convtranspose2d_params, EncoderSpec


class TrConv2dLayer(torch.nn.Module):
    """
    TrConv2dLayer is a 2D transposed convolution followed by batch normalization

    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param encoder_spec: EncoderSpec describing the corresponding encoder convolution to be inverted.
    :param activation: Optional non-linear activation function applied after batch normalization.
        Default is torch.nn.ReLU.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels:  int,
                 encoder_spec: EncoderSpec,
                 activation: Optional[Callable] = torch.nn.ReLU):
        super().__init__()

        if not isinstance(encoder_spec, EncoderSpec):
            raise ValueError(f"Invalid encoder spec: {encoder_spec}. Must be an instance of EncoderSpec.")


        self.in_channels = in_channels
        self.out_channels = out_channels

        params = compute_convtranspose2d_params(encoder_spec=encoder_spec)
        self.upsampling_kernel = params.upsampling_kernel
        self.upsampling_stride = params.upsampling_stride
        self.upsampling_padding = params.upsampling_padding

        self.deconv2d = torch.nn.ConvTranspose2d(in_channels=self.in_channels,
                                                out_channels=self.out_channels,
                                                kernel_size=self.upsampling_kernel,
                                                stride=self.upsampling_stride,
                                                padding=self.upsampling_padding)

        self.batchnorm2d = torch.nn.BatchNorm2d(num_features=self.out_channels)
        self.activation = activation(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv2d(x)
        x = self.batchnorm2d(x)
        x = self.activation(x)
        return x
