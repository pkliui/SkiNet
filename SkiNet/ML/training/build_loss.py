from typing import cast

import segmentation_models_pytorch as smp
import torch
from SkiNet.Utils.experiment_keys import LossFunctionKey


class BCEDiceLoss(torch.nn.Module):
    """Combined BCE and Dice loss with equal weighting."""

    def __init__(self) -> None:
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="binary")

    def forward(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, 0.5 * self.bce(logits, mask) + 0.5 * self.dice(logits, mask))  # cast for mypy


def build_loss(loss_key: LossFunctionKey) -> torch.nn.Module:
    """
    Build a loss function instance from a LossFunctionKey.

    :param loss_key: LossFunctionKey enum value identifying the loss function to build.
    :return: An instance of the loss function specified by the key.
    """
    if loss_key == LossFunctionKey.BCE:
        return torch.nn.BCEWithLogitsLoss()

    if loss_key == LossFunctionKey.DICE:
        return cast(torch.nn.Module, smp.losses.DiceLoss(mode="binary"))  # cast for mypy

    if loss_key == LossFunctionKey.BCE_DICE:
        return BCEDiceLoss()

    raise ValueError(
        f"Unsupported loss: {loss_key!r}. Supported values: {[k.value for k in LossFunctionKey]}"
    )
