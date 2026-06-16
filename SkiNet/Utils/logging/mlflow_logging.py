import logging
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from lightning.pytorch.callbacks import EarlyStopping

from SkiNet.Utils.mlops.mlflow_utils import _safe_log_mlflow_param
from SkiNet.ML.configs.train_configs.train_config import TrainConfig
from SkiNet.ML.configs.experiment_config import ExperimentConfig


logger = logging.getLogger(__name__)


def _log_mlflow_run_metadata(mlflow_logger: MLFlowLogger, main_config: ExperimentConfig) -> None:
    """
    Logs relevant metadata about the MLflow run, including framework, project, experiment type, model kind,
    dataset kind as well as the configuration file used for this run.

    :param mlflow_logger: The MLFlowLogger instance used for logging.
    :param main_config: The main configuration object loaded from the configuration file
    """
    model_cfg = main_config.modelconfig
    tags = {
        "framework": "lightning",
        "project": "SkiNet",
        "experiment_type": main_config.experiment_type,
        "model_kind": model_cfg.kind,
        "dataset_kind": main_config.dataconfig.kind,
        # Architecture tags — present on UNet2DModelConfig; guarded with hasattr so
        # future model types that lack these fields don't require changes here.
        "model/encoder_residual_mode": getattr(model_cfg, "encoder_residual_mode", None),
        "model/merge_residual_mode": getattr(model_cfg, "merge_residual_mode", None),
        "model/se_reduction": getattr(model_cfg, "se_reduction", None),
    }
    # log the tags to mlflow, converting None values to strings and skipping logging for tags with None values
    for key, value in tags.items():
        if value is not None:
            mlflow_logger.experiment.set_tag(mlflow_logger.run_id, key, str(value))
        else:
            logger.warning(f"MLflow run tag '{key}' has None value; skipping log for this tag.")
    #
    # log the config file as an artifact in mlflow
    if main_config.cfg_path:
        mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, main_config.cfg_path, artifact_path="config")
    else:
        logger.warning("No config file path found in main_config.cfg_path; skipping logging config artifact to MLflow.")


def _log_fit_and_optimizer_params_to_mlflow(mlflow_logger: MLFlowLogger, train_cfg: TrainConfig) -> None:
    """
    Log the training config params such as fitting and optimizer configuration to MLflow
    """
    params = {
        "fit/batch_size": train_cfg.batch_size,
        "fit/num_workers": train_cfg.num_workers,
        "optimizer/name": train_cfg.optimizer_name,
        "optimizer/lr": train_cfg.lr,
        "fit/max_epochs": train_cfg.max_epochs,
        "fit/accelerator": train_cfg.accelerator,
        "fit/devices": train_cfg.devices,
        "fit/log_every_n_steps": train_cfg.log_every_n_steps,
    }
    for key, value in params.items():
        _safe_log_mlflow_param(mlflow_logger, key, value)


def _log_early_stopping_config_to_mlflow(mlflow_logger: MLFlowLogger, early_stopping: EarlyStopping) -> None:
    """
    Log the config parameters of the EarlyStopping callback to MLflow as one-shot before training params
    """
    params = {
        "early_stopping/monitor": early_stopping.monitor,
        "early_stopping/mode": early_stopping.mode,
        "early_stopping/min_delta": early_stopping.min_delta,
        "early_stopping/patience": early_stopping.patience,
        "early_stopping/strict": early_stopping.strict,
        "early_stopping/check_finite": early_stopping.check_finite,
        "early_stopping/stopping_threshold": early_stopping.stopping_threshold,
        "early_stopping/divergence_threshold": early_stopping.divergence_threshold,
    }
    for key, value in params.items():
        _safe_log_mlflow_param(mlflow_logger, key, value)
