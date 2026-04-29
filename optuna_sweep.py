import argparse
from copy import deepcopy
import logging
from pathlib import Path
import mlflow
from typing import Callable, TypeAlias, cast
import optuna
from optuna.trial import Trial
from optuna.samplers import GridSampler

from SkiNet.ML.configs.load_config_from_yaml import load_config_from_yaml
from SkiNet.ML.configs.experiment_config import ExperimentConfig
from main_run import train_and_evaluate
from SkiNet.Utils.mlops.optuna_utils import scale_lr, validate_search_space
from SkiNet.Utils.experiment_keys import HyperparamKey

logger = logging.getLogger(__name__)

GridValue: TypeAlias = str | float | int | bool | None
SearchSpace: TypeAlias = dict[str, list[GridValue]]


def build_objective(main_config: ExperimentConfig, monitor: str, search_space: SearchSpace) -> Callable[[Trial], float]:
    """
    Create an Optuna objective that logs each trial as a nested MLflow child run.
    """

    def objective(trial: optuna.trial.Trial) -> float:
        """
        Optuna objective function for a single hyperparameter trial.

        The function samples hyperparameters from the search space, trains and evaluates the model,
        and logs all parameters and metrics to MLflow as a nested child run under the
        parent study run.

        The following hyperparameters are sampled via categorical suggestion:
            - lr: learning rate
            - weight_decay: L2 regularisation strength
            - batch_size: number of samples per training batch

        :param trial: Optuna Trial object used to sample hyperparameters and
                    report intermediate/final values back to the study.
        :return: The value of the monitored metric (e.g. ``val_dice``) for this
                trial, which Optuna uses to determine the direction of optimisation.
        :raises KeyError: If the monitored metric is not present in the metrics dict
                        returned by ``train_and_evaluate``, indicating a mismatch
                        between the ``--monitor`` argument and the keys logged by
                        the LightningModule.
        """
        cfg = deepcopy(main_config)
        train_cfg = cfg.trainconfig

        # define the search space targets and reassign the respective configs
        lr = cast(float, trial.suggest_categorical(HyperparamKey.LR, search_space[HyperparamKey.LR]))
        weight_decay = cast(float, trial.suggest_categorical(HyperparamKey.WEIGHT_DECAY, search_space[HyperparamKey.WEIGHT_DECAY]))
        batch_size = cast(int, trial.suggest_categorical(HyperparamKey.BATCH_SIZE, search_space[HyperparamKey.BATCH_SIZE]))

        # scale LR linearly with batch size, anchored to the smallest batch in the sweep
        # so sampled LR values always represent the rate at the sweep's reference batch size,
        # regardless of what TRAIN_CONFIG.batch_size is set to.
        BASE_BATCH_SIZE = min(cast(list[int], search_space[HyperparamKey.BATCH_SIZE]))
        scaled_lr = scale_lr(lr=lr, batch_size=batch_size, base_batch_size=BASE_BATCH_SIZE)
        train_cfg.lr = scaled_lr

        train_cfg.weight_decay = weight_decay
        train_cfg.batch_size = batch_size

        run_name = f"trial_{trial.number}_lr{lr}_wd{weight_decay}_bs{batch_size}"

        # Define a CHILD run for the current combination of hyperparameters
        with mlflow.start_run(run_name=run_name, nested=True) as child_run:
            # Tag the child to be able to search/filter by study
            mlflow.set_tag("optuna_trial", trial.number)
            mlflow.set_tag("optuna_study", cfg.sweepconfig.experiment_name)
            # Log all search space targets
            mlflow.log_param("lr_optuna_sampled", lr)       # what Optuna picked
            mlflow.log_param("lr_actually_used", scaled_lr)  # what the model actually used
            mlflow.log_param("weight_decay", weight_decay)
            mlflow.log_param("batch_size", batch_size)

            # Get the fit-time validation metrics
            metrics = train_and_evaluate(cfg, visualize=False)

            # Check if the monitor (metrics to optimise) is there
            if monitor not in metrics:
                available = ", ".join(sorted(metrics))
                raise KeyError(f"Monitor '{monitor}' not found in callback metrics. Available: {available}")

            score = metrics[monitor]
            mlflow.log_metric(monitor, score)
            mlflow.set_tag("child_run_id", child_run.info.run_id)

        return score

    return objective


def main() -> None:
    """
    Entry point for running an Optuna hyperparameter sweep over a grid search space.

    Loads an experiment config from a YAML file, sets up an MLflow parent run to
    wrap the entire study, then launches an Optuna GridSampler study where each
    trial is logged as a nested MLflow child run.

    The search space is fixed to:
        - lr: [3e-4, 1e-4]          # base LRs at min(batch_size) in the sweep; scaled linearly per trial
        - weight_decay: [1e-4, 1e-3]
        - batch_size: [16, 32]

    CLI arguments:
        --config      Path to the experiment YAML config file (required). This is the main_config.yaml file containing
            various parameters as set in ExperimentConfig class
        --trials      Number of Optuna trials to run (default: 24).
        --monitor     Metric to optimise, must match a key returned by
                      ``train_and_evaluate`` (default: ``val_best_dice_at_threshold``).
        --direction   Optimisation direction, either ``maximize`` or
                      ``minimize`` (default: ``maximize``).
        --experiment  MLflow experiment name under which the parent run is
                      registered (default: ``optuna_sweep``).

    :raises SystemExit: If Optuna is not installed in the current environment.

    Example using default number of trials:
      ```python
      python optuna_sweep.py --config main_config.yaml --monitor val_best_dice_at_threshold --direction maximize
      ```
    Example using a custom number of trials:
      ```python
      python optuna_sweep.py --config main_config.yaml --monitor val_best_dice_at_threshold --direction maximize --trials 10
      ```

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML config. "
                        "This is the main_config.yaml file containing various parameters as set in ExperimentConfig class")
    parser.add_argument("--trials", type=int, default=None,
                        help="Optional number of trials to run. Defaults to full grid (n_combos).")
    parser.add_argument("--monitor", type=str, default="val_best_dice_at_threshold", help="Metric to optimize")
    parser.add_argument("--direction", type=str, default="maximize", choices=["maximize", "minimize"])
    parser.add_argument("--experiment", type=str, default="optuna_sweep", help="MLflow experiment name")
    args = parser.parse_args()

    # load the main config
    main_config = load_config_from_yaml(args.config)

    # define the search space based on sweep config
    search_space = main_config.sweepconfig.search_space
    validate_search_space(search_space, set(HyperparamKey))

    # Set tracking URI from config FIRST, before any mlflow.start_run calls,
    # so the parent run and MLFlowLogger all land on the same backend.
    tracking_uri = main_config.trainconfig.mlflow_config.tracking_uri
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        logger.warning("MLflow tracking URI is not set in config. Using default local file storage. "
                       "Set 'trainconfig.mlflow_config.tracking_uri' to log to a remote server or different location.")
    mlflow.set_experiment(args.experiment)

    # Define the PARENT run that wraps the entire study — all trials appear as children beneath it
    with mlflow.start_run(run_name=f"optuna_study_{main_config.trainconfig.experiment_name}_{args.monitor}") as _:
        import math
        n_combos = math.prod(len(v) for v in search_space.values())
        n_trials = args.trials if args.trials is not None else n_combos

        if args.trials is not None and args.trials < n_combos:
            logger.warning(f"--trials ({args.trials}) < full grid size ({n_combos}). "
                           "GridSampler will cover the grid only partially with no coverage guarantee. "
                           "Use --trials without an argument to run the full grid.")

        mlflow.set_tag("mlflow.runName", f"optuna_study_{args.monitor}")
        mlflow.set_tag("monitor", args.monitor)
        mlflow.set_tag("direction", args.direction)
        mlflow.set_tag("n_trials", n_trials)
        if args.trials is not None:
            mlflow.log_param("requested_n_trials", args.trials)

        # set number of trials and combinations of search space params
        mlflow.log_param("n_combos", n_combos)
        mlflow.log_param("n_trials", n_trials)

        # create optuna study and begin the optimisation
        study = optuna.create_study(direction=args.direction,
                                    sampler=GridSampler(search_space,
                                                        seed=main_config.trainconfig.seed))
        study.optimize(build_objective(main_config, args.monitor, search_space), n_trials=n_trials)

        # BEST STUDY RESULTS:
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
