import time
from typing import Any
import lightning as L
import torch


class ThroughputCallback(L.Callback):
    """
    Logs training throughput metrics at the end of every training batch:
      - perf/samples_per_sec  — examples processed per second
      - perf/time_per_step_ms — wall-clock duration of the forward+backward+optimiser step (ms)

    These two metrics are the primary signals for the hardware batch-size feasibility sweep
      - If throughput doubles when batch size doubles, GPU is not yet saturated (perfect scaling).
      - If time_per_step grows super-linearly,  a bottleneck exists (e.g. CPU workers)

    Usage: add to the Lightning trainer callbacks list alongside SystemMetricsThreadCallback.
    """

    def on_train_batch_start(self,
                             trainer: L.Trainer,
                             pl_module: L.LightningModule,
                             batch: Any,
                             batch_idx: int) -> None:
        self._t0 = time.perf_counter()

    def on_train_batch_end(self,
                           trainer: L.Trainer,
                           pl_module: L.LightningModule,
                           outputs: Any,
                           batch: Any,
                           batch_idx: int) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._t0
        # trainer.train_dataloader may be a CombinedLoader wrapping the real DataLoader
        raw_loader = trainer.train_dataloader
        dl = getattr(raw_loader, "loaders", raw_loader)  # unwrap CombinedLoader if present
        batch_size = getattr(dl, "batch_size", None)
        if batch_size is None or elapsed <= 0:
            return
        pl_module.log("perf/time_per_step_ms", elapsed * 1000,
                      on_step=True, on_epoch=False, prog_bar=False, logger=True)
        pl_module.log("perf/samples_per_sec", batch_size / elapsed,
                      on_step=True, on_epoch=False, prog_bar=True, logger=True)
