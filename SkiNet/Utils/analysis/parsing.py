"""Reusable MLflow analysis helpers for segmentation experiments.

The functions in this module intentionally use only the MLflow SQLite store and
local artifact layout, so the same notebook can be reused by changing `db_path`
and `artifact_root`.
"""

from __future__ import annotations

from SkiNet.Utils.experiment_keys import NetworkBlockKey

ARCH_PATTERN = NetworkBlockKey.arch_pattern()


def parse_encoder_merge(experiment_name: str) -> tuple[str | None, str | None]:
    """
    Extract encoder and merge mode from experiment names

    :param experiment_name: experiment name to parse in the form `enc-x_merge-y`
    :return: encoder name, merge mode
    """
    match = ARCH_PATTERN.search(experiment_name)
    if not match:
        return None, None
    return match.group(1), match.group(2)
