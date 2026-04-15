from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import datetime
import logging
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import MLFlowLogger
from SkiNet.Utils.logging.system_metrics import SystemMetricsThreadCallback
from SkiNet.Utils.mlops.mlflow_callbacks import MLflowTrainingArtifactsCallback
from SkiNet.Utils.logging.mlflow_logging import (_log_mlflow_run_metadata,
                                                 _log_fit_and_optimizer_params_to_mlflow,
                                                 _log_early_stopping_config_to_mlflow)
from SkiNet.ML.configs.experiment_config import ExperimentConfig
from SkiNet.Utils.mlops.mlflow_utils import _ensure_mlflow_available

logger = logging.getLogger(__name__)


@dataclass
class TrainerComponents:
    """
    Encapsulate loggers and callbacks built for a training run

    """
    run_name: str
    loggers: list
    callbacks: list
    mlflow_logger: MLFlowLogger | None


def setup_logging_and_callbacks(*, main_config: ExperimentConfig) -> TrainerComponents:
    """
    Sets up loggers and callbacks based on the provided configuration.

    :param main_config: The main configuration object loaded from the configuration file
    :return: A LoggingSetup dataclass containing the run name, loggers, callbacks, mlflow logger,
    and early stopping callback (if used).
    """

    # initialize loggers and callbacks
    lightning_loggers: list = []
    lightning_callbacks: list = []
    early_stopping: EarlyStopping | None = None
    checkpoint_cb: ModelCheckpoint | None = None
    mlflow_logger: MLFlowLogger | None = None

    # read training configuration
    train_cfg = main_config.trainconfig
    # set run name with timestamp for uniqueness
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{train_cfg.experiment_name}_{timestamp}"

    # --- Callbacks for logging system metrics to available loggers  ---

    lightning_callbacks.append(SystemMetricsThreadCallback(interval_sec=train_cfg.system_metrics_interval_sec))

    # --- Callbacks for early stopping in Lightning  ---

    if train_cfg.use_early_stopping:
        early_stopping = EarlyStopping(**train_cfg.early_stopping_config.model_dump())
        lightning_callbacks.append(early_stopping)

    # --- Callbacks for saving model checkpoints  ---

    if train_cfg.use_checkpoint:
        # For segmentation, sometimes Dice/IoU is a better “best model” criterion than validation loss, depending on what you care about.
        # see mlflowconfig - currenly val/loss is monitored
        checkpoint_dir = (Path(train_cfg.log_dir) / "checkpoints" / run_name).resolve()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_cb = ModelCheckpoint(dirpath=str(checkpoint_dir),
                                        monitor=train_cfg.checkpoint_config.monitor,  # ok
                                        mode=train_cfg.checkpoint_config.mode,  # ok
                                        save_top_k=train_cfg.checkpoint_config.save_top_k,  # ok
                                        save_last=train_cfg.checkpoint_config.save_last,  # ok
                                        filename=train_cfg.checkpoint_config.filename,  # ok
                                        auto_insert_metric_name=False)
        lightning_callbacks.append(checkpoint_cb)

    # --- MLflow Loggers and callbacks ---

    if train_cfg.use_mlflow_logger:
        _ensure_mlflow_available(train_cfg.mlflow_config.tracking_uri,
                                 train_cfg.mlflow_config.fallback_to_local_mlflow)

        # set up mlflow logger
        mlflow_logger = MLFlowLogger(experiment_name=train_cfg.experiment_name,
                                     run_name=run_name,
                                     tracking_uri=train_cfg.mlflow_config.tracking_uri,
                                     log_model=train_cfg.mlflow_config.log_model)
        lightning_loggers.append(mlflow_logger)

        _log_mlflow_run_metadata(mlflow_logger=mlflow_logger,
                                 main_config=main_config)
        _log_fit_and_optimizer_params_to_mlflow(mlflow_logger=mlflow_logger,
                                                train_cfg=train_cfg)
        # runtime metrics and best checkpoint will be logged in the MLflowTrainingArtifactsCallback
        # gated on early stoppping and checkpoint inside
        lightning_callbacks.append(MLflowTrainingArtifactsCallback(mlflow_logger=mlflow_logger,
                                                                   log_model_summary=train_cfg.mlflow_config.log_model_summary,
                                                                   early_stopping_cb=early_stopping,
                                                                   checkpoint_cb=checkpoint_cb))
        if early_stopping:
            _log_early_stopping_config_to_mlflow(mlflow_logger, early_stopping)

    # --- Litlogger ---
    if train_cfg.use_litlogger_logger:
        import litlogger
        if not train_cfg.litlogger_config.teamspace:
            logger.warning("Litlogger is enabled but litlogger_teamspace is empty; metrics may not appear in dashboard.")
        lightning_loggers.append(litlogger.LightningLogger(name=run_name,
                                                           root_dir=train_cfg.log_dir,
                                                           teamspace=train_cfg.litlogger_config.teamspace,
                                                           log_model=train_cfg.litlogger_config.log_model,
                                                           save_logs=train_cfg.litlogger_config.save_logs,
                                                           checkpoint_name=train_cfg.litlogger_config.checkpoint_name))

    # --- Learning rate monitoring ---
    lightning_callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    return TrainerComponents(run_name=run_name,
                             loggers=lightning_loggers,
                             callbacks=lightning_callbacks,
                             mlflow_logger=mlflow_logger)
