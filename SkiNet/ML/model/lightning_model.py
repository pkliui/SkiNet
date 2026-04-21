import lightning as L
import torch
import logging
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex
from torchmetrics.functional.classification import binary_f1_score

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

        # Initialise lists to keep validation and ground truth probs for finding the optimal threshold
        self.optimal_threshold = 0.5  # start at 0.5, update each val epoch
        self._val_probs: list[torch.Tensor] = []
        self._val_masks: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.nn.Module:
        model: torch.nn.Module = self.model(x)
        return model

    def _get_probs_and_preds(self, logits: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Computes sigmoid activation function probabilities and predictions
        based on an optimal threshold value (for each epoch N computed based on params in epoch N-1),
        converts the masks into long.

        :param logits: outputs of the model
        :param mask: ground truth binary masks

        :return a dictionary of probabilities, predictions and the ground truth masks converted to long:
            - probabilities are the sigmoid activation function outputs,
            - predictions are the thresholded outputs of the sigmoid activation function, converted to long
            - inarise masks in case the values shifted from 0  and 1 due to interpolation in augmenations
        """
        probs = torch.sigmoid(logits)
        preds = (probs >= self.optimal_threshold).long()
        target = (mask >= 0.5).long()
        return {"probs": probs, "preds": preds, "target": target}

    def _compute_and_log_segmentation_metrics(self, prefix: str, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Compute and log Dice and IoU metrics and log it at the end of each epoch.
        It is called from every validation_step and test_step. Add any new torchmetrics here.

        :param prefix: "val" or "test"
        :param preds: predictions, which are the thresholded outputs of the sigmoid activation function, converted to long
        :param target: ground truth, which are the masks, converted to long
        """
        metrics = {
            f"{prefix}_dice": getattr(self, f"{prefix}_dice"),
            f"{prefix}_iou": getattr(self, f"{prefix}_iou"),
        }
        for name, metric in metrics.items():
            self.log(name, metric(preds, target), on_step=False, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=preds.shape[0])

    def _compute_and_log_threshold_search_metrics_for_sigmoid(self) -> None:
        """
        Compute and log Dice metrics by sweeping over different sigmoid thresholds,
        using probabilities from all batches saved during validation.

        Log the best threshold and the best Dice metrics.
        """
        if not self._val_probs:
            return

        # get probabilities from the whole validation set
        all_probs = torch.cat(self._val_probs)
        # get targets by converting to long and thresholding in case augmentations produced some interpolated values
        all_targets = (torch.cat(self._val_masks) >= 0.5).long()

        self._val_probs.clear()
        self._val_masks.clear()

        if torch.unique(all_targets).numel() < 2:
            return

        best_thr = 0.5
        best_dice = -1.0

        # .tolist() converts to plain Python floats, which PyTorch will broadcast to whatever device the tensor is on
        for thr in torch.linspace(0.0, 1.0, 51).tolist():
            preds = (all_probs >= thr).long()
            dice = binary_f1_score(preds, all_targets)
            if dice.item() > best_dice:
                best_dice = dice.item()  # keep as float throughout
                best_thr = thr  # already float thanks to tolist()

        self.optimal_threshold = best_thr
        self.log("val_optimal_threshold", self.optimal_threshold,
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_best_dice_at_threshold", best_dice, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def _prepare_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if isinstance(self.loss_fn, torch.nn.BCEWithLogitsLoss):
            if not torch.is_floating_point(mask):
                mask = mask.float()
            if mask.max() > 1:
                mask = mask.float() / 255.0
        return mask

    def _get_current_lr(self) -> float | None:
        if self.trainer is None or not self.trainer.optimizers:
            return None
        optimizer: torch.optim.Optimizer = self.trainer.optimizers[0]
        lr_current: float = optimizer.param_groups[0]["lr"]
        return lr_current

    @staticmethod
    def _tensor_debug_summary(name: str, tensor: torch.Tensor) -> str:
        detached = tensor.detach()
        finite_mask = torch.isfinite(detached)
        finite_count = int(finite_mask.sum().item())
        total_count = detached.numel()
        if finite_count == 0:
            return f"{name}: shape={tuple(detached.shape)}, dtype={detached.dtype}, finite=0/{total_count}"
        finite_values = detached[finite_mask]
        return (
            f"{name}: shape={tuple(detached.shape)}, dtype={detached.dtype}, "
            f"finite={finite_count}/{total_count}, min={float(finite_values.min().item()):.6g}, "
            f"max={float(finite_values.max().item()):.6g}"
        )

    def _raise_if_non_finite(self, name: str, tensor: torch.Tensor, batch_idx: int) -> None:
        if torch.isfinite(tensor).all():
            return
        raise ValueError(
            f"Non-finite values detected in batch {batch_idx}. "
            f"{self._tensor_debug_summary(name, tensor)}"
        )

    def _shared_eval_step(self, prefix: str, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        This is a shared function for validation and testing.
        Logs metrics and additionally saves validation probabilities
        for threshold search at the end of the validation epoch.
        """
        x = batch.get("image")
        mask = batch.get("mask")
        if x is None or mask is None:  # for mypy type checking
            raise ValueError(f"Batch is missing 'image' or 'mask' keys. Found keys: {list(batch.keys())}")
        mask = self._prepare_mask(mask)

        # get the logits and compute loss from them
        logits = self.model(x)
        loss: torch.Tensor = self.loss_fn(logits, mask)
        self.log(f"{prefix}_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=(prefix == "val"), logger=True, batch_size=x.shape[0])

        # get probabilities and predictions for metrics computation
        probs_and_preds = self._get_probs_and_preds(logits, mask)
        probs = probs_and_preds["probs"]
        preds = probs_and_preds["preds"]
        target = probs_and_preds["target"]

        # update epoch metrics from this batch; Lightning logs the aggregated value at epoch end
        self._compute_and_log_segmentation_metrics(prefix, preds, target)

        # collect probabilities (raw sigmoid outputs) and raw masks to find an optimal sigmoid threshold at the val epoch end
        # note we collect all batches - so full validation set will be used at the end of epoch
        if prefix == "val":
            self._val_probs.append(probs.detach().reshape(-1))
            self._val_masks.append(mask.detach().reshape(-1))
            self.log("val_threshold_used", self.optimal_threshold, on_step=False, on_epoch=True, logger=True)
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x = batch.get("image")
        mask = batch.get("mask")
        if x is None or mask is None:
            raise ValueError(f"Batch is missing 'image' or 'mask' keys. Found keys: {list(batch.keys())}")
        mask = self._prepare_mask(mask)
        self._raise_if_non_finite("train/image", x, batch_idx)
        self._raise_if_non_finite("train/mask", mask, batch_idx)

        logger.debug(f"Training step - batch_idx: {batch_idx}, input image shape: {x.shape}, mask shape: {mask.shape}")

        logits = self.model(x)
        self._raise_if_non_finite("train/logits", logits, batch_idx)
        t_loss: torch.Tensor = self.loss_fn(logits, mask)
        self._raise_if_non_finite("train/loss", t_loss, batch_idx)

        self.log("train_loss", t_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=x.shape[0])
        return t_loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        if self.optimizer_name == "adam":
            optim = torch.optim.Adam(self.parameters(), lr=self.lr, eps=1e-4)
        elif self.optimizer_name == "adamw":
            optim = torch.optim.AdamW(self.parameters(), lr=self.lr, eps=1e-4, weight_decay=self.weight_decay)
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

    def on_validation_epoch_end(self) -> None:
        """
        The following is executed on validation epoch end.
        Find for the optimal sigmoid activation function threshold
        that results in the highest Dice score.
        """
        self._compute_and_log_threshold_search_metrics_for_sigmoid()


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
