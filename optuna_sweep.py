import argparse
from copy import deepcopy
import logging
from pathlib import Path
import mlflow
from typing import Callable
from optuna.trial import Trial

from SkiNet.ML.configs.load_config_from_yaml import load_config_from_yaml
from SkiNet.ML.configs.experiment_config import ExperimentConfig
from main_run import train_and_evaluate

logger = logging.getLogger(__name__)


def build_objective(base_config: ExperimentConfig, monitor: str) -> Callable[[Trial], float]:
    """
    Create an Optuna objective that logs each trial as a nested MLflow child run.
    """
    import optuna

    def objective(trial: optuna.trial.Trial) -> float:
        cfg = deepcopy(base_config)
        train_cfg = cfg.trainconfig

        lr = trial.suggest_categorical("lr", [1e-3, 3e-4, 1e-4, 3e-5])
        weight_decay = trial.suggest_categorical("weight_decay", [1e-4, 1e-3])

        train_cfg.lr = lr
        train_cfg.weight_decay = weight_decay

        run_name = f"trial_{trial.number}_lr{lr}_wd{weight_decay}"

        with mlflow.start_run(run_name=run_name, nested=True) as child_run:
            # Tag the child so you can search/filter by study
            mlflow.set_tag("optuna_trial", trial.number)
            mlflow.set_tag("optuna_study", base_config.trainconfig.__class__.__name__)
            mlflow.log_param("lr", lr)
            mlflow.log_param("weight_decay", weight_decay)

            metrics = train_and_evaluate(cfg, visualize=False)

            if monitor not in metrics:
                available = ", ".join(sorted(metrics))
                raise KeyError(
                    f"Monitor '{monitor}' not found in callback metrics. Available: {available}"
                )

            score = metrics[monitor]
            mlflow.log_metric(monitor, score)
            mlflow.set_tag("child_run_id", child_run.info.run_id)

        return score

    return objective


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML config")
    parser.add_argument("--trials", type=int, default=6, help="Number of Optuna trials to run")
    parser.add_argument("--monitor", type=str, default="val_dice", help="Metric to optimize")
    parser.add_argument("--direction", type=str, default="maximize", choices=["maximize", "minimize"])
    parser.add_argument("--experiment", type=str, default="optuna_sweep", help="MLflow experiment name")
    args = parser.parse_args()

    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Optuna is not installed. Install it before using optuna_sweep.py."
        ) from exc

    base_config = load_config_from_yaml(args.config)

    # Set tracking URI from config FIRST, before any mlflow.start_run calls,
    # so the parent run and MLFlowLogger all land on the same backend.
    tracking_uri = base_config.trainconfig.mlflow_config.tracking_uri
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(args.experiment)
    else:
        logger.warning(
            "MLflow tracking URI is not set in config. Using default local file storage. "
            "Set 'trainconfig.mlflow_config.tracking_uri' to log to a remote server or different location."
        )

    # Parent run wraps the entire study — all trials appear as children beneath it
    with mlflow.start_run(run_name=f"optuna_study_{args.monitor}") as _:
        mlflow.set_tag("mlflow.runName", f"optuna_study_{args.monitor}")
        mlflow.set_tag("monitor", args.monitor)
        mlflow.set_tag("direction", args.direction)
        mlflow.set_tag("n_trials", args.trials)
        mlflow.log_param("n_trials", args.trials)

        study = optuna.create_study(direction=args.direction)
        study.optimize(build_objective(base_config, args.monitor), n_trials=args.trials)

        # Log best results on the parent run for easy comparison
        mlflow.log_metric(f"best_{args.monitor}", study.best_value)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.set_tag("best_trial", study.best_trial.number)

    print(f"Best trial value ({args.monitor}): {study.best_value}")
    print("Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
