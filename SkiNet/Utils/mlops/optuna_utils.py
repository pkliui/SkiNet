import torch
import lightning as L
import logging
import math
from collections.abc import Set
from typing import List
from SkiNet.Utils.experiment_keys import HyperparamKey

logger = logging.getLogger(__name__)


def _collect_trainer_metrics(light_trainer: L.Trainer) -> dict[str, float]:
    """
    Collect scalar metrics from all trainer metric stores.
    Used in HPO to return a flat dict of all scalar metrics after fit/test for Optuna to log and optimize on.
    """
    out: dict[str, float] = {}
    metric_sources = (light_trainer.callback_metrics,
                      light_trainer.logged_metrics,
                      light_trainer.progress_bar_metrics)
    for source in metric_sources:
        for key, value in source.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    scalar = float(value.detach().cpu().item())
                else:
                    logger.warning("Skipping metric '%s': expected scalar tensor, got shape %s", key, value.shape)
                    continue
            elif isinstance(value, (float, int)):
                scalar = float(value)
            else:
                continue

            if key in out and not math.isclose(out[key], scalar, rel_tol=1e-6):
                raise ValueError(
                    f"Conflicting values for metric '{key}' across trainer metric stores: "
                    f"{out[key]} vs {scalar}."
                )
            out[key] = scalar
    return out


def scale_lr(lr: float, batch_size: int, base_batch_size: int) -> float:
    """
    Scale learning rate linearly with batch size.

    :param lr: Learning rate calibrated at ``base_batch_size``
    :param batch_size: Target batch size for this trial
    :param base_batch_size: Reference batch size at which ``lr`` is calibrated
    :return: Scaled learning rate
    """
    if base_batch_size <= 0:
        raise ValueError(f"base_batch_size must be positive, got {base_batch_size}")
    return lr * (batch_size / base_batch_size)


def validate_search_space(
    search_space: dict[str, List[int | float]],
    expected_keys: Set[str] | None = None,
) -> None:
    """
    Validate search space parameters against the keys the objective function reads.

    :param search_space: Search space parameters
    :param expected_keys: Keys the objective function reads from search_space.
        Defaults to the full ``HyperparamKey`` set used by the Optuna sweep.
    :raises ValueError: If search space keys do not match expected_keys
    """
    if expected_keys is None:
        expected_keys = set(HyperparamKey)

    actual_keys = set(search_space.keys())
    if actual_keys != expected_keys:
        unexpected = actual_keys - expected_keys
        missing = expected_keys - actual_keys
        raise ValueError(
            f"search_space keys do not match what objective reads. "
            f"Unexpected: {unexpected}, missing: {missing}"
        )
