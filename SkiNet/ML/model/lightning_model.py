import lightning as L
import torch
import logging
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex

from SkiNet.ML.configs.experiment_config import ExperimentConfig
from SkiNet.ML.configs.train_configs.train_config import ReduceOnPlateauConfig
from SkiNet.ML.model.model_factory import create_model

logger = logging.getLogger(__name__)


class LightningModel(L.LightningModule):
    def __init__(self,
                 model: torch.nn.Module,
                 loss_fn: torch.nn.Module,
                 num_classes: int,
                 lr: float,
                 optimizer_name: str,
                 weight_decay: float,
                 lr_scheduler_config: ReduceOnPlateauConfig):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "loss_fn"])
        self.model = model
        self.loss_fn = loss_fn
        self.num_classes = num_classes
        self.lr = lr
        self.optimizer_name = optimizer_name.lower()
        self.weight_decay = weight_decay
        self.lr_scheduler_config = lr_scheduler_config
        # Metrics for validation and testing
        self.val_dice = BinaryF1Score()
        self.val_iou = BinaryJaccardIndex()
        self.test_dice = BinaryF1Score()
        self.test_iou = BinaryJaccardIndex()

    def forward(self, x: torch.Tensor) -> torch.nn.Module:
        model: torch.nn.Module = self.model(x)
        return model

    def _get_preds(self, logits: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns float logits target for loss, and long preds/target for metrics."""
        preds = (torch.sigmoid(logits) >= 0.5).long()
        target = (mask >= 0.5).long()
        return preds, target

    def _log_segmentation_metrics(self, prefix: str, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Logs Dice and IoU. Add any new torchmetrics here."""
        metrics = {
            f"{prefix}_dice": getattr(self, f"{prefix}_dice"),
            f"{prefix}_iou": getattr(self, f"{prefix}_iou"),
        }
        for name, metric in metrics.items():
            self.log(name, metric(preds, target), on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def _prepare_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if isinstance(self.loss_fn, torch.nn.BCEWithLogitsLoss):
            if not torch.is_floating_point(mask):
                mask = mask.float()
            if mask.max() > 1:
                mask /= 255.0
        return mask

    def _get_current_lr(self) -> float | None:
        if self.trainer is None or not self.trainer.optimizers:
            return None
        optimizer: torch.optim.Optimizer = self.trainer.optimizers[0]
        lr_current: float = optimizer.param_groups[0]["lr"]
        return lr_current

    def _shared_eval_step(self, prefix: str, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch.get("image")
        mask = batch.get("mask")
        if x is None or mask is None:  # for mypy type checking
            raise ValueError(f"Batch is missing 'image' or 'mask' keys. Found keys: {list(batch.keys())}")
        mask = self._prepare_mask(mask)

        logits = self.model(x)
        preds, target = self._get_preds(logits, mask)
        loss: torch.Tensor = self.loss_fn(logits, mask)

        self.log(f"{prefix}_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=(prefix == "val"), logger=True, batch_size=x.shape[0])
        self._log_segmentation_metrics(prefix, preds, target)
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x = batch.get("image")
        mask = batch.get("mask")
        if x is None or mask is None:
            raise ValueError(f"Batch is missing 'image' or 'mask' keys. Found keys: {list(batch.keys())}")
        mask = self._prepare_mask(mask)

        logger.debug(f"Training step - batch_idx: {batch_idx}, input image shape: {x.shape}, mask shape: {mask.shape}")

        logits = self.model(x)
        t_loss: torch.Tensor = self.loss_fn(logits, mask)

        self.log("train_loss", t_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=x.shape[0])
        return t_loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        if self.optimizer_name == "adam":
            optim = torch.optim.Adam(self.parameters(), lr=self.lr, eps=1e-8)
        elif self.optimizer_name == "adamw":
            optim = torch.optim.AdamW(self.parameters(), lr=self.lr, eps=1e-8, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer_name '{self.optimizer_name}'. Supported: ['adam', 'adamw']")

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim,
                                                               mode=self.lr_scheduler_config.mode,
                                                               patience=self.lr_scheduler_config.patience,
                                                               factor=self.lr_scheduler_config.factor)
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.lr_scheduler_config.monitor,
            },
        }

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_eval_step("val", batch)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_eval_step("test", batch)


def build_lightning_model(main_config: ExperimentConfig) -> LightningModel:
    """
    Build the Lightning segmentation model from the experiment config.
    """
    train_cfg = main_config.trainconfig

    model: torch.nn.Module = create_model(main_config)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    return LightningModel(model=model,
                          loss_fn=loss_fn,
                          num_classes=1,
                          lr=train_cfg.lr,
                          optimizer_name=train_cfg.optimizer_name,
                          weight_decay=train_cfg.weight_decay,
                          lr_scheduler_config=train_cfg.lr_scheduler_config)
