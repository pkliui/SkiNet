from typing import Any
from lightning.pytorch.loggers.mlflow import MLFlowLogger
import os
import logging
import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


def _safe_log_mlflow_param(mlflow_logger: MLFlowLogger, key: str, value: Any) -> None:
    """
    Logs a single parameter to MLflow, converting it to a string and handling None values by skipping the log.

    :param mlflow_logger: The MLFlowLogger instance used for logging.
    :param key: The parameter name to log.
    :param value: The parameter value to log. If None, the parameter will not be logged.
    """
    if value is None:
        logger.debug(f"MLflow param '{key}' is None, skipping.")
        return
    try:
        mlflow_logger.experiment.log_param(mlflow_logger.run_id, key, str(value))
    except Exception as e:
        logger.warning(f"Failed to log MLflow param '{key}': {e}")


def _ensure_mlflow_available(tracking_uri: str | None, fallback_to_local_mlflow: bool = False) -> None:
    """
    Make sure the MLflow tracking server is reachable at the specified tracking URI.
    If not, optionally fall back to a local file-based tracking URI.

    :param tracking_uri: The MLflow tracking URI to check for availability.
    :param fallback_to_local_mlflow: If True, will set the tracking URI to a local file store if the original URI is unreachable.
    """
    client = MlflowClient(tracking_uri=tracking_uri)  # probe the same URI you'll actually use for logging
    try:
        # quick probe: list_experiments will raise on unreachable HTTP endpoint
        client.search_experiments()
    except Exception as exc:
        if not fallback_to_local_mlflow:
            raise RuntimeError(
                "MLflow tracking server appears unreachable and fallback to local MLflow is disabled. "
                "Please ensure the tracking server is running and MLFLOW_TRACKING_URI is set correctly."
            ) from exc
        fallback_dir = os.path.join(os.getcwd(), "mlruns")
        fallback_uri = f"file:{fallback_dir}"
        logger.warning(
            "MLflow tracking server appears unreachable (%s). Falling back to local tracking URI: %s",
            exc, fallback_uri
        )
        os.environ["MLFLOW_TRACKING_URI"] = fallback_uri
        mlflow.set_tracking_uri(fallback_uri)
