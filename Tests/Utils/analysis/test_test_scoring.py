"""Unit tests for SkiNet.Utils.analysis.test_scoring."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest
import torch

from SkiNet.Utils.analysis.test_scoring import (
    bootstrap_ci,
    build_ckpt_map,
    per_image_dice_iou,
    score_at_thresholds,
)

# Two images, 4 pixels each. At thr=0.5: image 0 is a perfect match (Dice 1),
# image 1 predicts one false-positive pixel against an empty mask (Dice ~0).
_PROBS = torch.tensor([[0.9, 0.8, 0.2, 0.1],
                       [0.9, 0.2, 0.2, 0.1]])
_MASKS = torch.tensor([[1.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0]])


# ---------------------------------------------------------------------------
# per_image_dice_iou
# ---------------------------------------------------------------------------

class TestPerImageDiceIou:
    def test_returns_one_score_per_image(self) -> None:
        dice, iou = per_image_dice_iou(_PROBS, _MASKS, 0.5)
        assert dice.shape == (2,)
        assert iou.shape == (2,)

    def test_perfect_and_wrong_images(self) -> None:
        dice, iou = per_image_dice_iou(_PROBS, _MASKS, 0.5)
        assert dice[0] == pytest.approx(1.0, abs=1e-4)   # exact overlap
        assert iou[0] == pytest.approx(1.0, abs=1e-4)
        assert dice[1] == pytest.approx(0.0, abs=1e-4)   # FP against empty mask

    def test_threshold_changes_predictions(self) -> None:
        # At thr=0.95 nothing fires; image 1 (empty mask, empty pred) -> Dice ~1.
        dice, _ = per_image_dice_iou(_PROBS, _MASKS, 0.95)
        assert dice[1] == pytest.approx(1.0, abs=1e-4)

    def test_is_per_image_not_pooled(self) -> None:
        # Per-image mean (0.5) differs from a pooled-pixel Dice, which would weight
        # the two images' pixels together. This guards the ISIC averaging.
        dice, _ = per_image_dice_iou(_PROBS, _MASKS, 0.5)
        assert dice.mean() == pytest.approx(0.5, abs=1e-4)


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

class TestBootstrapCi:
    def test_lo_le_hi_and_brackets_mean(self) -> None:
        values = np.array([0.80, 0.82, 0.84, 0.86, 0.88])
        lo, hi = bootstrap_ci(values, n_boot=500, seed=0)
        assert lo <= hi
        assert lo <= values.mean() <= hi

    def test_deterministic_with_seed(self) -> None:
        values = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
        assert bootstrap_ci(values, 200, seed=7) == bootstrap_ci(values, 200, seed=7)

    def test_zero_width_for_constant_values(self) -> None:
        values = np.full(10, 0.83)
        lo, hi = bootstrap_ci(values, 100, seed=1)
        assert lo == pytest.approx(0.83)
        assert hi == pytest.approx(0.83)


# ---------------------------------------------------------------------------
# score_at_thresholds
# ---------------------------------------------------------------------------

class TestScoreAtThresholds:
    def test_one_row_per_threshold(self) -> None:
        out = score_at_thresholds(_PROBS, _MASKS, [0.5, 0.95], n_boot=100, seed=0)
        assert list(out["threshold"]) == [0.5, 0.95]
        assert set(out.columns) == {"threshold", "dice", "iou", "dice_lo", "dice_hi"}

    def test_per_image_mean_dice_matches(self) -> None:
        out = score_at_thresholds(_PROBS, _MASKS, [0.5], n_boot=100, seed=0)
        assert out.loc[0, "dice"] == pytest.approx(0.5, abs=1e-4)

    def test_ci_brackets_dice(self) -> None:
        out = score_at_thresholds(_PROBS, _MASKS, [0.5], n_boot=300, seed=0)
        r = out.iloc[0]
        assert r["dice_lo"] <= r["dice"] <= r["dice_hi"]


# ---------------------------------------------------------------------------
# build_ckpt_map
# ---------------------------------------------------------------------------

def _make_db(path: Path, seed_uuids: dict[int, str]) -> None:
    """Minimal MLflow store mapping run names (seedNNN) to uuids."""
    with closing(sqlite3.connect(path)) as con:
        con.execute("CREATE TABLE runs (run_uuid TEXT, name TEXT)")
        con.executemany(
            "INSERT INTO runs VALUES (?, ?)",
            [(uuid, f"run_seed{seed}") for seed, uuid in seed_uuids.items()],
        )
        con.commit()


class TestBuildCkptMap:
    def test_maps_seed_to_ckpt_via_uuid(self, tmp_path: Path) -> None:
        uuids = {100: "a" * 32, 108: "b" * 32}
        db = tmp_path / "sweep.db"
        _make_db(db, uuids)
        for seed, uuid in uuids.items():
            ck = tmp_path / "mlruns" / "1" / uuid / "artifacts" / "checkpoints" / "best"
            ck.mkdir(parents=True)
            (ck / f"epoch{seed}.ckpt").touch()

        pattern = str(tmp_path / "mlruns" / "1" / "*" / "artifacts" / "checkpoints" / "best" / "*.ckpt")
        ckpt_map = build_ckpt_map(db, glob_pattern=pattern, project_root=tmp_path)
        assert sorted(ckpt_map) == [100, 108]
        assert ckpt_map[108].name == "epoch108.ckpt"

    def test_ignores_uuid_with_no_seed_run(self, tmp_path: Path) -> None:
        db = tmp_path / "sweep.db"
        _make_db(db, {100: "a" * 32})
        orphan = tmp_path / "mlruns" / "1" / ("c" * 32) / "artifacts" / "checkpoints" / "best"
        orphan.mkdir(parents=True)
        (orphan / "epoch9.ckpt").touch()
        pattern = str(tmp_path / "mlruns" / "1" / "*" / "artifacts" / "checkpoints" / "best" / "*.ckpt")
        ckpt_map = build_ckpt_map(db, glob_pattern=pattern, project_root=tmp_path)
        assert ckpt_map == {}

    def test_multi_db_merges_seeds(self, tmp_path: Path) -> None:
        """Seeds from two separate MLflow DBs are merged into one map."""
        db1, db2 = tmp_path / "a.db", tmp_path / "b.db"
        _make_db(db1, {100: "a" * 32})
        _make_db(db2, {200: "b" * 32})
        for seed, uuid in [(100, "a" * 32), (200, "b" * 32)]:
            ck = tmp_path / "mlruns" / uuid / "best"
            ck.mkdir(parents=True)
            (ck / f"seed{seed}.ckpt").touch()
        pattern = str(tmp_path / "mlruns" / "*" / "best" / "*.ckpt")
        ckpt_map = build_ckpt_map(db1, db2, glob_pattern=pattern, project_root=tmp_path)
        assert sorted(ckpt_map) == [100, 200]

    def test_run_name_without_seed_is_skipped(self, tmp_path: Path) -> None:
        """A run whose name contains no seedNNN produces no entry."""
        db = tmp_path / "sweep.db"
        with closing(sqlite3.connect(db)) as con:
            con.execute("CREATE TABLE runs (run_uuid TEXT, name TEXT)")
            con.execute("INSERT INTO runs VALUES (?, ?)", ("a" * 32, "no_seed_here"))
            con.commit()
        ck = tmp_path / "mlruns" / ("a" * 32) / "best"
        ck.mkdir(parents=True)
        (ck / "epoch0.ckpt").touch()
        pattern = str(tmp_path / "mlruns" / "*" / "best" / "*.ckpt")
        ckpt_map = build_ckpt_map(db, glob_pattern=pattern, project_root=tmp_path)
        assert ckpt_map == {}


# ---------------------------------------------------------------------------
# per_image_dice_iou  — additional edge cases
# ---------------------------------------------------------------------------

class TestPerImageDiceIouEdges:
    def test_single_image(self) -> None:
        probs = torch.tensor([[0.9, 0.1]])
        masks = torch.tensor([[1.0, 0.0]])
        dice, iou = per_image_dice_iou(probs, masks, 0.5)
        assert dice.shape == (1,)
        assert dice[0] == pytest.approx(1.0, abs=1e-4)

    def test_all_positive_mask_perfect_prediction(self) -> None:
        """All pixels foreground, model predicts all foreground → Dice = IoU = 1."""
        probs = torch.ones(1, 6) * 0.9
        masks = torch.ones(1, 6)
        dice, iou = per_image_dice_iou(probs, masks, 0.5)
        assert dice[0] == pytest.approx(1.0, abs=1e-4)
        assert iou[0] == pytest.approx(1.0, abs=1e-4)

    def test_all_positive_mask_empty_prediction(self) -> None:
        """All pixels foreground, model predicts nothing → Dice ~0."""
        probs = torch.zeros(1, 6)
        masks = torch.ones(1, 6)
        dice, iou = per_image_dice_iou(probs, masks, 0.5)
        assert dice[0] == pytest.approx(0.0, abs=1e-4)

    def test_returns_numpy_not_tensor(self) -> None:
        dice, iou = per_image_dice_iou(_PROBS, _MASKS, 0.5)
        assert isinstance(dice, np.ndarray)
        assert isinstance(iou, np.ndarray)


# ---------------------------------------------------------------------------
# score_at_thresholds — edge cases
# ---------------------------------------------------------------------------

class TestScoreAtThresholdsEdges:
    def test_empty_threshold_list_returns_empty_df(self) -> None:
        # pd.DataFrame([]) produces an empty frame with no columns — that's the
        # current contract; this test pins the behaviour so a future change is visible.
        out = score_at_thresholds(_PROBS, _MASKS, [], n_boot=10, seed=0)
        assert len(out) == 0

    def test_iou_le_dice_for_imperfect_prediction(self) -> None:
        """IoU ≤ Dice is a mathematical identity (same numerator, larger denominator)."""
        out = score_at_thresholds(_PROBS, _MASKS, [0.5], n_boot=50, seed=0)
        assert out.loc[0, "iou"] <= out.loc[0, "dice"] + 1e-6

    def test_single_threshold_repeated_is_deterministic(self) -> None:
        out1 = score_at_thresholds(_PROBS, _MASKS, [0.5], n_boot=200, seed=42)
        out2 = score_at_thresholds(_PROBS, _MASKS, [0.5], n_boot=200, seed=42)
        assert out1.equals(out2)
