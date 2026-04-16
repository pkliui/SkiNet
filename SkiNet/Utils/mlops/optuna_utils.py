import torch
import lightning as L


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
                    out[key] = float(value.detach().cpu().item())
            elif isinstance(value, (float, int)):
                out[key] = float(value)
    return out
