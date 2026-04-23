import torch
import lightning as L
import logging
import math
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


def scale_lr(lr: float, batch_size: int, base_batch_size: int = 16) -> float:
    """
    Scale learning rate linearly with batch size.

    :param lr: Learning rate
    :param batch_size: Batch size
    :param base_batch_size: Base batch size (default: 16)
    :return: Scaled learning rate
    """
    if base_batch_size <= 0:
        raise ValueError(f"base_batch_size must be positive, got {base_batch_size}")
    return lr * (batch_size / base_batch_size)
