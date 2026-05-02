import logging
from dataclasses import dataclass
from typing import cast

import torch.nn as nn
from torch import Tensor

from SkiNet.ML.model.architecture.base_segmentation import BaseSegmentation
from SkiNet.ML.model.blocks.conv2d_layer import Conv2dLayer
from SkiNet.ML.model.blocks.decoder2d import Decoder2D
from SkiNet.ML.model.blocks.encoder2d import Encoder2D
from SkiNet.ML.model.blocks.merge2d_block import Merge2DBlock
from SkiNet.ML.utils.model_utils import initialise_weights
from SkiNet.ML.utils.sampling.decoder_sampling import get_decoder_params_2d
from SkiNet.ML.utils.sampling.encoder_sampling import get_encoder_params_2d
from SkiNet.ML.utils.typing_utils import IntOrTuple2d

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EncoderPath:
    """
    Value object encapsulating the encoder path components.

    :param encoders: Sequential encoder layers (shallow  -> deep).
    :param out_channels: Final output channel count for the last encoder layer.
    """
    encoders: nn.ModuleList
    out_channels: int


@dataclass(frozen=True)
class DecoderPath:
    """
    Immutable value object encapsulating the decoder path.

    **Invariants** (enforced at construction):
    1. len(decoders) == len(mergeblocks) — paired 1:1
    2. For each (decoder, mergeblock) pair:
       - decoder.layer_number == mergeblock.layer_number — structural alignment
       - mergeblock.layer_number >= 2
    3. out_channels matches the output channel count of the shallowest decoder layer

    :param decoders: Sequential decoder layers (deep -> shallow).
    :param mergeblocks: Skip connection merge blocks aligned with decoders.
    :param out_channels: Final output channel count for the last decoder layer.
    """
    decoders: nn.ModuleList
    mergeblocks: nn.ModuleList
    out_channels: int

    def __post_init__(self) -> None:
        # Invariant 1: Pairing count
        if len(self.decoders) != len(self.mergeblocks):
            err = "Decoder count %s != merge block count %s. Decoders and merge blocks must be paired 1:1." % (
                len(self.decoders), len(self.mergeblocks))
            logger.error(err)
            raise ValueError(err)

        # Invariant 2: Layer number alignment
        for i, (dec_mod, merge_mod) in enumerate(zip(self.decoders, self.mergeblocks)):
            dec = cast(Decoder2D, dec_mod)  # for mypy
            merge = cast(Merge2DBlock, merge_mod)  # for mypy
            if dec.layer_number < 2 or merge.layer_number < 2:
                err = "Decoders and merge blocks must be at layers 2..N. "\
                    "Decoder layer number=%s or merge block layer number=%s." % (dec.layer_number, merge.layer_number)
                logger.error(err)
                raise ValueError(err)

            if dec.layer_number != merge.layer_number:
                err = "Layer number mismatch at index "\
                    "%s: decoder.layer_number=%s != merge.layer_number=%s" % (i, dec.layer_number, merge.layer_number)
                logger.error(err)
                raise ValueError(err)

        # Invariant 3: Output channel consistency (if decoders is non-empty)
        if len(self.decoders) > 0:
            shallowest_decoder_out = self.decoders[-1].out_channels
            if self.out_channels != shallowest_decoder_out:
                err = "Output channel mismatch: "\
                    "DecoderPath.out_channels=%s != shallowest_decoder.out_channels=%s" % (
                        self.out_channels, shallowest_decoder_out)
                logger.error(err)
                raise ValueError(err)


class UNet2D(BaseSegmentation):
    """
    UNet 2D model

    :param in_channels: Number of input channels.
    :param out_channels_layer1: Number of output channels in the 1st layer of the encoder.
    :param kernel: Kernel size of the convolution operation. Default is 3.
    :param stride: Stride of the convolution operation. If not 1, it acts as a downsampling factor in encoder layers and as
        an upsampling factor in decoder layers. Default is 2.
    :param dilation: Dilation factor of the convolution operation. Default is 1.
    :param number_of_layers: Number of layers in the encoder path. Default is 5.
        The count starts from layer 1, which is the shallowest layer.
        The number of decoder layers is number_of_layers - 1 and there is one more additional last convolutional layer.
    :param num_output_classes: Number of output classes for segmentation. Default is 1.
    :param model_name: Name of the model.
    :param validate_forward: If True, perform validation checks on skip connections during the forward pass. Default is False.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels_layer1: int,
                 kernel: IntOrTuple2d = 3,
                 stride: IntOrTuple2d = 2,
                 dilation: IntOrTuple2d = 1,
                 number_of_layers: int = 5,
                 num_output_classes: int = 1,
                 model_name: str = "UNet2D",
                 validate_forward: bool = False) -> None:

        super().__init__()

        self.in_channels = in_channels
        self.out_channels_layer1 = out_channels_layer1
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.number_of_layers = number_of_layers
        self.num_output_classes = num_output_classes
        self.model_name = model_name
        self.validate_forward = validate_forward

        self.layer1_params = get_encoder_params_2d(kernel=self.kernel, stride=1, dilation=self.dilation)
        """As encoder's layer 1 is not downsampling, compute its convolution parameters separately"""
        self.params = get_encoder_params_2d(kernel=self.kernel, stride=self.stride, dilation=self.dilation)
        """Encoder's convolution parameters for layers starting from layer 2"""
        self.decoder_params = get_decoder_params_2d(self.params)
        """Decoder's convolution parameters, applies to layers N through 2"""

        self._build_unet()
        self.apply(initialise_weights)

    def _build_unet(self) -> None:
        """
        Build all layers of the UNet.
        """
        encoder_path = self._build_encoders()
        self.encoders = encoder_path.encoders

        decoder_path = self._build_decoder_path(encoder_path.out_channels)
        self.decoders = decoder_path.decoders
        self.mergeblocks = decoder_path.mergeblocks

        self.last_layer = self._build_last_layer(decoder_path.out_channels)

    def _build_encoders(self) -> EncoderPath:
        """
        Build the encoder path of the UNet.
        Precondition: number_of_layers must be >=2.

        The encoder path consists of `number_of_layers` sequential Encoder2D blocks.
        - The first encoder layer does not perform downsampling.
        - Each subsequent encoder layer downsamples spatially and doubles the number of channels.

        :return EncoderPath.encoders: A list of Encoder2D modules ordered from shallow to deep.
        :return EncoderPath.out_channels: Number of output channels produced by the deepest encoder layer.
            This value is used to initialize the decoder path.
        """
        encoders = nn.ModuleList()

        in_channels = self.in_channels
        out_channels = self.out_channels_layer1

        # 1st encoder layer - no downsampling
        encoders.append(Encoder2D(in_channels=in_channels,
                                  out_channels=out_channels,
                                  conv_params=self.layer1_params,
                                  apply_bias=False,
                                  activation=nn.ReLU,
                                  use_residual=True,
                                  layer_number=1))

        # each subsequent encoder layer is downsampling by stride and doubles the number of channels at the output
        for layer_number in range(2, self.number_of_layers+1):
            # update the number of channels for the next encoder layer
            in_channels = out_channels
            out_channels = in_channels * 2
            encoders.append(Encoder2D(in_channels=in_channels,
                                      out_channels=out_channels,
                                      conv_params=self.params,
                                      apply_bias=False,
                                      activation=nn.ReLU,
                                      use_residual=True,
                                      layer_number=layer_number))

        # the number of channels passed to the deepest decoder's layer is out_channels, the number of channels output by the deepest encoder layer
        return EncoderPath(encoders=encoders,
                           out_channels=out_channels)

    def _build_decoder_path(self, in_channels: int) -> DecoderPath:
        """
        Build the decoder path of the UNet, which consists of decoder layers Decoder2D and merge blocks Merge2DBlock for skip connections.

        Decoder structure:
        - There are `number_of_layers - 1` decoder blocks, corresponding to encoder layers N..2 (deep -> shallow).
        - Each decoder block upsamples spatially (according to `stride`) and reduces (typically halves) the channel count.

        Skip connections:
        - Skip feature maps are taken from encoder layers 1..N-1 (all encoder layers except the deepest one).
        - At decoder layer k (k = N..2), the decoder output is merged with the skip tensor from encoder layer k-1.
          Example for N=5:
            - decoder layer 5 merges with encoder layer 4 (deepest skip)
            - decoder layer 2 merges with encoder layer 1 (shallowest skip)

        :return DecoderPath.decoders: A list of Decoder2D modules ordered from deep to shallow.
        :return DecoderPath.mergeblocks: A list of Merge2DBlock modules aligned with the decoder layers.
        :return DecoderPath.out_channels: Number of output channels produced by the shallowest decoder layer.
            The value is provided to the network's last layer
        """
        decoders = nn.ModuleList()
        mergeblocks = nn.ModuleList()

        # each subsequent decoder layer is upsampling by stride and halves the number of channels at the output
        for (idx, layer_number) in enumerate(range(self.number_of_layers, 1, -1)):
            out_channels = in_channels // 2
            decoders.append(Decoder2D(layer_number=layer_number,
                                      in_channels=in_channels,
                                      out_channels=out_channels,
                                      decoder_params=self.decoder_params,
                                      activation=nn.ReLU))
            mergeblocks.append(Merge2DBlock(layer_number=layer_number,
                                            in_channels_from_decoder=out_channels,
                                            in_channels_from_skip=out_channels,
                                            out_channels=out_channels,
                                            conv_params=self.params,
                                            activation=nn.ReLU))
            # update the number of channels for the next decoder layer
            in_channels = out_channels

        # the number of channels passed to the last network's layer is the number of channels output by the shallowest decoder layer
        return DecoderPath(decoders=decoders,
                           mergeblocks=mergeblocks,
                           out_channels=in_channels)

    def _build_last_layer(self, in_channels: int) -> Conv2dLayer:
        """
        Build the final convolutional layer of the UNet.

        This layer maps the output of the last decoder layer to the desired number of output classes using a 1x1 convolution.

        :param in_channels: Number of input channels into the last layer, corresponds to the number of channels produced by the shallowest decoder layer.
        :return: Final convolutional layer producing the network output.
        """
        return Conv2dLayer(in_channels=in_channels,
                           out_channels=self.num_output_classes,
                           kernel=(1, 1),
                           stride=(1, 1),
                           dilation=(1, 1),
                           padding=(0, 0),
                           apply_batchnorm=False,
                           apply_bias=True,
                           activation=None)

    def _validate_skip_keys(self, skip_connections_dict: dict[int, Tensor]) -> None:
        """
        Check the keys in skip connections dict match the expected layer numbers for skip connections,
        which are 1..N-1 where N is the total number of layers in the encoder

        :param skip_connections_dict: dict mapping encoder layer numbers to skip tensors
        """
        # number_of_layers is not included, as the deepest encoder layer is not used for skip connections
        expected_keys = set(range(1, self.number_of_layers))

        actual_keys = set(skip_connections_dict.keys())

        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        if missing or extra:
            err = ("Skip keys mismatch. missing=%s, extra=%s, expected=%s, actual=%s" %
                   (sorted(missing), sorted(extra), sorted(expected_keys), sorted(actual_keys)))
            logger.error(err)
            raise ValueError(err)

    def _validate_skip_count(self, skip_connections_dict: dict[int, Tensor]) -> None:
        """
        Check that the number of skip connections matches the number of decoder layers.

        :param skip_connections_dict: dict mapping encoder layer numbers to skip tensors
        """
        if len(skip_connections_dict) != len(self.decoders):
            err = ("Skip count %s != decoder count %s" % (len(skip_connections_dict), len(self.decoders)))
            logger.error(err)
            raise ValueError(err)

    def _log_near_zero_skips(self, skip_connections_dict: dict[int, Tensor]) -> None:
        """
        Log warnings for skip connections with near-zero magnitude.

        :param skip_connections_dict: dict mapping encoder layer numbers to skip tensors
        """
        for layer_num, skip_tensor in skip_connections_dict.items():
            if skip_tensor.abs().mean() < 1e-6:
                logger.warning("Layer %s: Skip connection has near-zero magnitude", layer_num)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the network.
        """
        # Keep skip connections maps from encoder path
        skip_connections_dict: dict[int, Tensor] = {}

        # Encoder path
        for enc_mod in self.encoders:
            enc = cast(Encoder2D, enc_mod)  # for mypy
            x = enc(x)
            if enc.layer_number < self.number_of_layers:
                skip_connections_dict[enc.layer_number] = x

        if self.validate_forward:
            self._validate_skip_keys(skip_connections_dict)
            self._validate_skip_count(skip_connections_dict)
            self._log_near_zero_skips(skip_connections_dict)

        # Decoder path with skip connections
        for dec_mod, merge_mod in zip(self.decoders, self.mergeblocks):
            dec = cast(Decoder2D, dec_mod)  # for mypy
            merge = cast(Merge2DBlock, merge_mod)  # for mypy
            x = dec(x)
            x = merge(x, skip_connections_dict[dec.layer_number - 1])

        return cast(Tensor, self.last_layer(x))
