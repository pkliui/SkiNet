from typing import cast
import lightning as L
import torch
import logging
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex
from torchmetrics.functional.classification import binary_f1_score

from SkiNet.ML.configs.experiment_config import ExperimentConfig
from SkiNet.ML.configs.train_configs.train_config import CosineAnnealingConfig, ReduceOnPlateauConfig
from SkiNet.ML.model.model_factory import create_model
from SkiNet.ML.training.build_loss import build_loss
from SkiNet.ML.training.training_utils import find_best_threshold

logger = logging.getLogger(__name__)


class LightningModel(L.LightningModule):
    def __init__(self,
                 model: torch.nn.Module,
                 loss_fn: torch.nn.Module,
                 lr: float,
                 optimizer_name: str,
                 weight_decay: float,
                 lr_scheduler_config: ReduceOnPlateauConfig,
                 cosine_annealing_config: CosineAnnealingConfig,
                 scheduler_type: str = "reduce_on_plateau",
                 use_lr_scheduler: bool = True,
                 optimal_threshold: float | None = None):
        """
        :param model: backbone segmentation network (returns raw logits)
        :param loss_fn: loss function applied to logits and binary float masks
        :param lr: initial learning rate
        :param optimizer_name: "adam" or "adamw"
        :param weight_decay: L2 regularisation coefficient
        :param lr_scheduler_config: ReduceLROnPlateau scheduler settings
        :param optimal_threshold: fixed sigmoid threshold to use instead of sweeping.
            When None (default), the threshold is found via grid search each validation epoch.
            If specified, it is assumed that it is the optimal value found previously in
            the threshold sweep or any other regular value, e.g. 0.5.
            The threshold is used to obtain masks' predictions out of probabilities
            and in computation of Dice metrics.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model", "loss_fn"])
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.optimizer_name = optimizer_name.lower()
        self.weight_decay = weight_decay
        self.lr_scheduler_config = lr_scheduler_config
        self.cosine_annealing_config = cosine_annealing_config
        self.scheduler_type = scheduler_type
        self.use_lr_scheduler = use_lr_scheduler

        # Optimal threshold - Register as a buffer so that checkpoints contain the threshold value at the best epoch
        # Float attributes are invisible to the checkpoint system
        # start at 0.5, update each val epoch
        self.optimal_threshold: torch.Tensor
        self.register_buffer("optimal_threshold", torch.tensor(0.5))
        self._fixed_optimal_threshold = optimal_threshold

        # Train mode metrics
        self.train_dice = BinaryF1Score()
        self.train_iou = BinaryJaccardIndex()
        # Metrics for validation and testing
        self.val_dice = BinaryF1Score()
        self.val_iou = BinaryJaccardIndex()
        self.test_dice = BinaryF1Score()
        self.test_iou = BinaryJaccardIndex()

        # Initialise lists to keep validation and ground truth probs for finding the optimal threshold
        self._val_probs: list[torch.Tensor] = []
        self._val_masks: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the backbone and return raw logits (pre-sigmoid)."""
        return self.model(x)  # type: ignore[no-any-return]

    @staticmethod
    def _get_probs_and_preds(logits: torch.Tensor,
                             threshold: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Apply sigmoid to logits and threshold the result into binary predictions.

        :param logits: raw model outputs (pre-sigmoid)
        :param threshold: scalar tensor; sigmoid outputs >= threshold are predicted positive
        :return: dict with keys:
            - "probs": sigmoid outputs in [0, 1]
            - "preds": thresholded predictions as torch.long
        """
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).long()
        return {"probs": probs, "preds": preds}

    def _compute_and_log_segmentation_metrics(self, prefix: str, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Compute Dice and IoU for one batch and log them at epoch end. Add new torchmetrics here.

        :param prefix: "train", "val", or "test" — prepended to each metric name
        :param preds: thresholded binary predictions as torch.long
        :param target: ground truth binary masks as torch.long
        """
        metrics = {
            f"{prefix}_dice": getattr(self, f"{prefix}_dice"),
            f"{prefix}_iou": getattr(self, f"{prefix}_iou"),
        }
        for name, metric in metrics.items():
            self.log(name, metric(preds, target), on_step=False, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=preds.shape[0], sync_dist=True)

    def _compute_and_log_threshold_search_metrics_for_sigmoid(self) -> None:
        """
        Compute and log Dice metrics by sweeping over different sigmoid thresholds,
        using probabilities from all batches saved during validation.

        Log the best threshold and the best Dice metrics.

        If self._fixed_optimal_threshold is not None, then use this value as the optimal threshold
        and to compute Dice metrics (for experiments with a fixed threshold value)
        """
        if not self._val_probs:
            return

        # get probabilities from the whole validation set
        all_probs = torch.cat(self._val_probs)
        # get targets by converting to long and thresholding in case augmentations produced some interpolated values
        all_targets = (torch.cat(self._val_masks) >= 0.5).long()

        self._val_probs.clear()
        self._val_masks.clear()

        # single-class edge case — log sentinel value so the metric key always exists
        # this prevents KeyError in Optuna when val split contains only one class
        if torch.unique(all_targets).numel() < 2:
            logger.warning("Validation targets contain only one class — skipping threshold sweep. "
                           "Logging val_best_dice_at_threshold=0.0 as sentinel.")
            self.log("val_best_dice_at_threshold", 0.0, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            return

        fixed_thr_preds = (all_probs >= 0.5).long()
        fixed_thr_dice = binary_f1_score(fixed_thr_preds, all_targets).item()

        if self._fixed_optimal_threshold is not None:
            best_thr = self._fixed_optimal_threshold
            best_preds = (all_probs >= best_thr).long()
            best_dice = binary_f1_score(best_preds, all_targets).item()
        else:
            best_result = find_best_threshold(all_probs, all_targets)
            best_thr, best_dice = best_result["best_threshold"], best_result["best_dice"]

        self.optimal_threshold.fill_(best_thr)
        self.log("val_optimal_threshold", self.optimal_threshold,
                 on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_best_dice_at_threshold", best_dice, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_dice_threshold_gain", best_dice - fixed_thr_dice,
                 on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

    @staticmethod
    def _prepare_mask(mask: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(mask):
            mask = mask.float()
        return (mask >= 0.5).float()

    @staticmethod
    def _tensor_debug_summary(name: str, tensor: torch.Tensor) -> str:
        """
        Build a diagnostic string for a tensor showing shape, dtype, finite element count,
        and min/max over finite elements (omitted when no finite values exist).

        :param name: label to prefix the summary with (e.g. "train/logits")
        :param tensor: tensor to summarise
        :return: human-readable one-line summary string
        """
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
        """
        Raise ValueError if the tensor contains any NaN or Inf value.

        :param name: descriptive label included in the error message (e.g. "train/logits")
        :param tensor: tensor to check
        :param batch_idx: current batch index, included in the error message for reproducibility
        :raises ValueError: if any element is non-finite
        """
        if torch.isfinite(tensor).all():
            return
        raise ValueError(
            f"Non-finite values detected in batch {batch_idx}. "
            f"{self._tensor_debug_summary(name, tensor)}"
        )

    def _shared_eval_step(self, prefix: str, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
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
        self._raise_if_non_finite(f"{prefix}/image", x, batch_idx)
        self._raise_if_non_finite(f"{prefix}/mask", mask, batch_idx)

        # get the logits and compute loss from them
        logits = self.model(x)
        self._raise_if_non_finite(f"{prefix}/logits", logits, batch_idx)
        loss: torch.Tensor = self.loss_fn(logits, mask)
        self._raise_if_non_finite(f"{prefix}/loss", loss, batch_idx)
        self.log(f"{prefix}_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=(prefix == "val"), logger=True, batch_size=x.shape[0], sync_dist=True)

        probs = self._compute_and_log_segmentation_metrics_from_logits_and_mask(prefix, logits, mask)

        # collect probabilities (raw sigmoid outputs) and raw masks to find an optimal sigmoid threshold at the val epoch end
        # note we collect all batches - so full validation set will be used at the end of epoch
        if prefix == "val":
            self._val_probs.append(probs.detach().cpu().reshape(-1))
            self._val_masks.append(mask.detach().cpu().reshape(-1))
            self.log("val_threshold_used", self.optimal_threshold, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Run one training iteration: validate inputs, compute loss, log metrics.

        :param batch: dict with "image" and "mask" tensors
        :param batch_idx: index of the current batch within the epoch
        :return: scalar training loss
        :raises ValueError: if the batch is missing required keys or contains non-finite values
        """
        x = batch.get("image")
        mask = batch.get("mask")
        if x is None or mask is None:
            raise ValueError(f"Batch is missing 'image' or 'mask' keys. Found keys: {list(batch.keys())}")
        mask = self._prepare_mask(mask)
        logger.debug(f"Training step - batch_idx: {batch_idx}, input image shape: {x.shape}, mask shape: {mask.shape}")

        # get the logits and compute and log loss
        logits = self.model(x)
        t_loss: torch.Tensor = self.loss_fn(logits, mask)

        # NaN/Inf checks force CUDA synchronisation on every step — restrict to warm-up steps only.
        if self.trainer.global_step < 3:
            self._raise_if_non_finite("train/image", x, batch_idx)
            self._raise_if_non_finite("train/mask", mask, batch_idx)
            self._raise_if_non_finite("train/logits", logits, batch_idx)
            self._raise_if_non_finite("train/loss", t_loss, batch_idx)
        self.log("train_loss", t_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=x.shape[0], sync_dist=True)

        _ = self._compute_and_log_segmentation_metrics_from_logits_and_mask("train", logits, mask)

        return t_loss

    def _compute_and_log_segmentation_metrics_from_logits_and_mask(
            self, prefix: str, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute and log segmentation metrics for one batch.

        :param prefix: "train", "val", or "test" — prepended to every logged metric name
        :param logits: raw model outputs (pre-sigmoid)
        :param mask: binary float mask from _prepare_mask; cast to long here for metrics
        :return: sigmoid probabilities (used upstream to accumulate val probs for threshold search)
        """
        target = mask.long()
        probs_and_preds = self._get_probs_and_preds(logits, self.optimal_threshold)
        self._compute_and_log_segmentation_metrics(prefix, probs_and_preds["preds"], target)
        return probs_and_preds["probs"]

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        Build the optimizer and optionally a scheduler.

        Scheduler type is controlled by ``scheduler_type``:
        - ``"reduce_on_plateau"``: ReduceLROnPlateau (requires a monitor metric)
        - ``"cosine_annealing"``: CosineAnnealingLR (T_max defaults to trainer.max_epochs)

        The scheduler is omitted entirely when use_lr_scheduler=False.

        :return: optimizer alone, or Lightning-compatible dict with "optimizer"
                 and "lr_scheduler" keys when the scheduler is enabled
        :raises ValueError: if optimizer_name is not "adam" or "adamw"
        :raises ValueError: if scheduler_type is not recognised
        """
        if self.optimizer_name == "adam":
            optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == "adamw":
            optim = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer_name '{self.optimizer_name}'. Supported: ['adam', 'adamw']")

        if not self.use_lr_scheduler:
            return cast(OptimizerLRScheduler, optim)

        if self.scheduler_type == "reduce_on_plateau":
            scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | torch.optim.lr_scheduler.CosineAnnealingLR
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optim,
                mode=self.lr_scheduler_config.mode,
                patience=self.lr_scheduler_config.patience,
                factor=self.lr_scheduler_config.factor,
            )
            return cast(OptimizerLRScheduler, {"optimizer": optim,
                                               "lr_scheduler": {"scheduler": scheduler,
                                                                "monitor": self.lr_scheduler_config.monitor}})

        if self.scheduler_type == "cosine_annealing":
            t_max: int = self.cosine_annealing_config.T_max or self.trainer.max_epochs or 1
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optim,
                T_max=t_max,
                eta_min=self.cosine_annealing_config.eta_min,
            )
            return cast(OptimizerLRScheduler, {"optimizer": optim,
                                               "lr_scheduler": {"scheduler": scheduler,
                                                                "interval": "epoch"}})

        raise ValueError(
            f"Unsupported scheduler_type '{self.scheduler_type}'. "
            "Supported: ['reduce_on_plateau', 'cosine_annealing']"
        )

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Log the gradient scaler attribute.
        If sufficiently high (e.g. >=1024), no gradient clipping required,
        if it monotonically decaying to 1, add clipping

        scaler.get_scale() forces a CUDA sync — only log every 50 steps to avoid
        serialising the async backward+optimizer pipeline on every step.
        """
        if self.trainer.global_step % 50 != 0:
            return
        scaler = getattr(self.trainer.precision_plugin, "scaler", None)
        if scaler is not None:
            self.log("grad_scale", scaler.get_scale(), prog_bar=True, on_step=True)

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Run one validation iteration and accumulate probabilities and masks for
        end-of-epoch threshold search.

        :param batch: dict with "image" and "mask" tensors
        :param batch_idx: index of the current batch within the epoch
        :return: scalar validation loss
        """
        return self._shared_eval_step("val", batch, batch_idx)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Test step using self.optimal_threshold learned during validation
        """
        return self._shared_eval_step("test", batch, batch_idx)

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
    if train_cfg.use_torch_compile:
        model = cast(torch.nn.Module, torch.compile(model))
        logger.info("torch.compile enabled — first forward pass will be slow (compilation)")
    loss_fn = build_loss(train_cfg.loss_name)
    return LightningModel(model=model,
                          loss_fn=loss_fn,
                          lr=train_cfg.lr,
                          optimizer_name=train_cfg.optimizer_name,
                          weight_decay=train_cfg.weight_decay,
                          lr_scheduler_config=train_cfg.lr_scheduler_config,
                          cosine_annealing_config=train_cfg.cosine_annealing_config,
                          scheduler_type=train_cfg.scheduler_type,
                          use_lr_scheduler=train_cfg.use_lr_scheduler,
                          optimal_threshold=train_cfg.optimal_threshold)
