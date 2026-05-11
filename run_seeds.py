"""
Run the same experiment with multiple seeds, optionally sweeping over encoder/merge mode combinations.

Single-mode (original behaviour) — all seeds run under one MLflow experiment:
    python run_seeds.py --config main_config.yaml --seeds 42 123 256

Ablation sweep — all 9 combinations × all seeds, each combination in its own MLflow experiment:
    python run_seeds.py --config main_config.yaml --seeds 42 123 256 \\
        --encoder-modes local_refinement he2 se \\
        --merge-modes   local_refinement he2 attention_gate

When --encoder-modes / --merge-modes are omitted the values in the YAML are used, so existing
call-sites are unaffected.  The MLflow experiment_name always encodes the active modes
(e.g. "ablation_model_unet2d-ph2_enc-he2_merge-attention_gate") so runs are unambiguous.
"""
import argparse
import copy
import logging
import re
from itertools import product
from pathlib import Path

import yaml

from SkiNet.ML.configs.load_config_from_yaml import load_config_from_yaml
from SkiNet.ML.configs.experiment_config import ExperimentConfig
from main_run import train_and_evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_VALID_ENCODER_MODES = ["local_refinement", "he2", "se"]
_VALID_MERGE_MODES = ["local_refinement", "he1", "he2", "attention_gate"]


def _patch_yaml_dict(yaml_dict: dict,
                     seed: int,
                     experiment_name: str,
                     *,
                     encoder_mode: str | None = None,
                     merge_mode: str | None = None) -> dict:
    """
    Return a deep copy of yaml_dict with seed, experiment_name, and optionally
    encoder/merge modes patched in.  Passing None for a mode leaves the YAML value intact.
    """
    d = copy.deepcopy(yaml_dict)
    d["TRAIN_CONFIG"]["seed"] = seed
    d["DATA_CONFIG"]["split_random_seed"] = seed
    d["TRANSFORM_CONFIG"]["seed_value"] = seed
    d["TRAIN_CONFIG"]["experiment_name"] = experiment_name
    if encoder_mode is not None:
        d["MODEL_CONFIG"]["encoder_residual_mode"] = encoder_mode
    if merge_mode is not None:
        d["MODEL_CONFIG"]["merge_residual_mode"] = merge_mode
    return d


def _config_from_dict(yaml_dict: dict, yaml_path: Path) -> ExperimentConfig:
    """Re-use the existing loader logic but from an already-patched dict.

    cfg_path is left pointing at the temp file (which contains the patched values)
    so MLflow uploads the actual config used for each run, not the base YAML.
    The temp file is not deleted here — MLflow reads it during artifact logging,
    after which the OS will clean it up on next reboot / tmpfs flush.
    """
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(yaml_dict, tmp)
        tmp_path = Path(tmp.name)
    cfg = load_config_from_yaml(tmp_path)
    cfg.cfg_path = str(tmp_path)  # keep patched file, not original main_config.yaml
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run multi-seed experiments, optionally sweeping encoder/merge modes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", type=Path, required=True, help="Path to base YAML config")
    ap.add_argument("--seeds", type=int, nargs="+", required=True, help="Seeds to run")
    ap.add_argument(
        "--name-prefix",
        type=str,
        default=None,
        help="Base experiment name (default: experiment_name from YAML with trailing _seed<N> stripped). "
             "Mode strings are always appended automatically.",
    )
    ap.add_argument(
        "--encoder-modes",
        nargs="+",
        choices=_VALID_ENCODER_MODES,
        default=None,
        metavar="MODE",
        help=f"Encoder modes to sweep. Choices: {_VALID_ENCODER_MODES}. "
             "Defaults to the single value in the YAML.",
    )
    ap.add_argument(
        "--merge-modes",
        nargs="+",
        choices=_VALID_MERGE_MODES,
        default=None,
        metavar="MODE",
        help=f"Merge modes to sweep. Choices: {_VALID_MERGE_MODES}. "
             "Defaults to the single value in the YAML.",
    )
    args = ap.parse_args()

    with open(args.config) as f:
        base_yaml = yaml.safe_load(f)

    # Derive base name from YAML if not provided, stripping a trailing _seed<N>.
    if args.name_prefix is None:
        raw_name: str = base_yaml["TRAIN_CONFIG"].get("experiment_name", "experiment")
        # Also strip any trailing _enc-* or _merge-* so re-runs don't accumulate suffixes.
        raw_name = re.sub(r"_enc-\w+", "", raw_name)
        raw_name = re.sub(r"_merge-\w+", "", raw_name)
        raw_name = re.sub(r"_seed\d+$", "", raw_name)
        args.name_prefix = raw_name

    # Fall back to YAML values when the caller did not specify modes, preserving
    # backward compatibility with existing single-mode call-sites.
    enc_modes: list[str] = args.encoder_modes or [
        base_yaml["MODEL_CONFIG"].get("encoder_residual_mode", "local_refinement")
    ]
    merge_modes: list[str] = args.merge_modes or [
        base_yaml["MODEL_CONFIG"].get("merge_residual_mode", "he2")
    ]

    combos = list(product(enc_modes, merge_modes))
    logger.info("Ablation plan: %d combination(s) × %d seed(s) = %d total run(s)",
                len(combos), len(args.seeds), len(combos) * len(args.seeds))

    all_metrics: dict[tuple[str, str, int], dict] = {}

    for enc, merge in combos:
        # Each (enc, merge) combination gets its own MLflow experiment so runs
        # are unambiguous in the UI even if the study is interrupted and resumed.
        experiment_name = f"{args.name_prefix}_enc-{enc}_merge-{merge}"

        logger.info("=" * 70)
        logger.info("Combination: encoder=%s  merge=%s  → experiment: %s", enc, merge, experiment_name)
        logger.info("=" * 70)

        for seed in args.seeds:
            logger.info("  Starting seed=%d", seed)
            patched = _patch_yaml_dict(base_yaml,
                                       seed=seed,
                                       experiment_name=experiment_name,
                                       encoder_mode=enc,
                                       merge_mode=merge)
            cfg = _config_from_dict(patched, args.config)
            metrics = train_and_evaluate(cfg, visualize=False)
            all_metrics[(enc, merge, seed)] = metrics
            logger.info("  seed=%d  enc=%s  merge=%s  →  %s", seed, enc, merge, metrics)

    logger.info("=" * 70)
    logger.info("All runs complete. Summary:")
    for (enc, merge, seed), metrics in all_metrics.items():
        logger.info("  enc=%-18s  merge=%-18s  seed=%-6d  %s", enc, merge, seed, metrics)


if __name__ == "__main__":
    main()
