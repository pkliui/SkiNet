from pathlib import Path
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, EarlyStopping
from lightning.pytorch.utilities.model_summary.model_summary import summarize
import torch
import lightning as L


class MLflowTrainingArtifactsCallback(Callback):
    """
    Logs training artifacts to MLflow:
    - model architecture summary as a text artifact at fit start
    - early stopping runtime metrics (best_score, wait_count, triggered) at fit end
    - best model checkpoint as an artifact at fit end
    """

    def __init__(self,
                 mlflow_logger: MLFlowLogger,
                 log_model_summary: bool = True,
                 early_stopping_cb: EarlyStopping | None = None,
                 checkpoint_cb: ModelCheckpoint | None = None) -> None:
        super().__init__()
        self.mlflow_logger = mlflow_logger
        self.log_model_summary = log_model_summary
        self.early_stopping_cb = early_stopping_cb
        self.checkpoint_cb = checkpoint_cb

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not self.log_model_summary:
            return
        summary_text = str(summarize(pl_module, max_depth=-1))
        self.mlflow_logger.experiment.log_text(self.mlflow_logger.run_id,
                                               summary_text,
                                               "model/model_summary.txt")

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._log_early_stopping_runtime_metrics(trainer)
        self._log_best_checkpoint()
        self._log_final_metrics(trainer)

    def _log_early_stopping_runtime_metrics(self, trainer: L.Trainer) -> None:
        """
        Logs early stopping runtime metrics such as best_score, wait_count, stopped_epoch and whether early stopping
        was triggered to MLflow if the early stopping callback is available.
        """
        if self.early_stopping_cb is None:
            return

        # use local variables with consistent types to satisfy static type checking
        raw_best = self.early_stopping_cb.best_score
        best_score_value: float | None
        if isinstance(raw_best, torch.Tensor):
            best_score_value = float(raw_best.detach().float().cpu().item())
        elif raw_best is None:
            best_score_value = None
        else:
            try:
                best_score_value = float(raw_best)  # handle numeric types
            except Exception:
                best_score_value = None

        if best_score_value is not None and best_score_value != float("inf") and best_score_value != float("-inf"):
            self.mlflow_logger.experiment.log_metric(self.mlflow_logger.run_id,
                                                     "early_stopping/best_score",
                                                     float(best_score_value),
                                                     step=int(trainer.global_step))

        # wait_count and stopped_epoch are numeric; coerce to float explicitly
        self.mlflow_logger.experiment.log_metric(self.mlflow_logger.run_id,
                                                 "early_stopping/wait_count",
                                                 float(self.early_stopping_cb.wait_count),
                                                 step=int(trainer.global_step))
        self.mlflow_logger.experiment.log_metric(self.mlflow_logger.run_id,
                                                 "early_stopping/stopped_epoch",
                                                 float(self.early_stopping_cb.stopped_epoch),
                                                 step=int(trainer.global_step))

        self.stopped_by_early_stopping = bool(self.early_stopping_cb.stopped_epoch > 0)
        self.mlflow_logger.experiment.log_metric(self.mlflow_logger.run_id,
                                                 "early_stopping/triggered",
                                                 1.0 if self.stopped_by_early_stopping else 0.0,
                                                 step=int(trainer.global_step))

    def _log_best_checkpoint(self) -> None:
        """
        Log the best model checkpoint as an artifact in MLflow if the checkpoint callback is available and has a best model path.
        """
        if self.checkpoint_cb is None or not self.checkpoint_cb.best_model_path:
            return
        best_checkpoint_path = Path(self.checkpoint_cb.best_model_path)
        if best_checkpoint_path.exists():
            self.mlflow_logger.experiment.log_artifact(self.mlflow_logger.run_id,
                                                       str(best_checkpoint_path),
                                                       artifact_path="checkpoints/best")

    def _log_final_metrics(self, trainer: L.Trainer) -> None:
        """
        Captures end-of-training summary metrics from the trainer and logs them to MLflow
        """
        for key, value in trainer.callback_metrics.items():
            # coerce tensor or numeric values into a float for logging
            numeric_value: float | None
            if isinstance(value, torch.Tensor):
                numeric_value = float(value.detach().float().cpu().item())
            elif isinstance(value, (float, int)):
                numeric_value = float(value)
            else:
                numeric_value = None

            if numeric_value is not None:
                self.mlflow_logger.experiment.log_metric(self.mlflow_logger.run_id,
                                                         f"final/{key}",
                                                         numeric_value,
                                                         step=int(trainer.global_step))
