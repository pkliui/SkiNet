"""
Run the same experiment with multiple seeds sequentially.

Usage:
    python run_seeds.py --config main_config.yaml --seeds 42 200 300
    python run_seeds.py --config main_config.yaml --seeds 42 200 300 --name-prefix unet2d_ph2
"""
import argparse
import copy
import logging
from pathlib import Path

import yaml

from SkiNet.ML.configs.load_config_from_yaml import load_config_from_yaml
from SkiNet.ML.configs.experiment_config import ExperimentConfig
from main_run import train_and_evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _patch_yaml_dict(yaml_dict: dict, seed: int, name_prefix: str) -> dict:
    d = copy.deepcopy(yaml_dict)
    d["TRAIN_CONFIG"]["seed"] = seed
    d["DATA_CONFIG"]["split_random_seed"] = seed
    d["TRANSFORM_CONFIG"]["seed_value"] = seed
    d["TRAIN_CONFIG"]["experiment_name"] = name_prefix  # same experiment for all seeds
    return d


def _config_from_dict(yaml_dict: dict, yaml_path: Path) -> ExperimentConfig:
    """Re-use the existing loader logic but from an already-patched dict."""
    # Write to a temp file so load_config_from_yaml can validate and construct normally.
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(yaml_dict, tmp)
        tmp_path = Path(tmp.name)
    try:
        cfg = load_config_from_yaml(tmp_path)
        cfg.cfg_path = str(yaml_path.resolve())
        return cfg
    finally:
        tmp_path.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True, help="Path to base YAML config")
    ap.add_argument("--seeds", type=int, nargs="+", required=True, help="Seeds to run")
    ap.add_argument(
        "--name-prefix",
        type=str,
        default=None,
        help="Experiment name prefix (default: taken from experiment_name in TRAIN_CONFIG with seed stripped)",
    )
    args = ap.parse_args()

    with open(args.config) as f:
        base_yaml = yaml.safe_load(f)

    # Derive prefix from existing name if not provided, stripping a trailing _seed<N> if present.
    if args.name_prefix is None:
        raw_name: str = base_yaml["TRAIN_CONFIG"].get("experiment_name", "experiment")
        import re
        args.name_prefix = re.sub(r"_seed\d+$", "", raw_name)

    all_metrics: dict[int, dict] = {}

    for seed in args.seeds:
        logger.info("=" * 60)
        logger.info("Starting run with seed=%d  (name: %s_seed%d)", seed, args.name_prefix, seed)
        logger.info("=" * 60)

        patched = _patch_yaml_dict(base_yaml, seed=seed, name_prefix=args.name_prefix)
        cfg = _config_from_dict(patched, args.config)

        metrics = train_and_evaluate(cfg, visualize=False)
        all_metrics[seed] = metrics
        logger.info("Seed %d finished. Metrics: %s", seed, metrics)

    logger.info("=" * 60)
    logger.info("All seeds done. Summary:")
    for seed, metrics in all_metrics.items():
        logger.info("  seed=%-6d  %s", seed, metrics)


if __name__ == "__main__":
    main()
