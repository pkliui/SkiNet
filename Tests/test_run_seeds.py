"""
Unit tests for the two pure helper functions in run_seeds.py.

train_and_evaluate is not called — these tests cover logic that runs
before any training, where bugs silently corrupt experiment configs.
"""
import copy
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# run_seeds.py lives at the repo root, not inside a package.
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# run_seeds.py uses a broken absolute import path ("repos.SkiNet.SkiNet…")
# that only works when the parent of the workspace root is on sys.path.
# Stub the missing module so we can test the pure helpers without a full env.
_stub_nbk = MagicMock()
_stub_nbk.ENC_PREFIX.value = "enc-"
_stub_nbk.MERGE_PREFIX.value = "merge-"
sys.modules.setdefault("repos", MagicMock())
sys.modules.setdefault("repos.SkiNet", MagicMock())
sys.modules.setdefault("repos.SkiNet.SkiNet", MagicMock())
sys.modules.setdefault("repos.SkiNet.SkiNet.Utils", MagicMock())
sys.modules.setdefault("repos.SkiNet.SkiNet.Utils.experiment_keys",
                       MagicMock(NetworkBlockKey=_stub_nbk))
# Also stub the training entry-point so importing run_seeds doesn't trigger ML deps.
sys.modules.setdefault("main_run", MagicMock())

from run_seeds import _patch_yaml_dict, _config_from_dict  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def base_yaml() -> dict:
    return {
        "TRAIN_CONFIG": {
            "seed": 0,
            "experiment_name": "ablation_model_unet2d-ph2_enc-he2_merge-attention_gate",
            "lr": 1e-4,
        },
        "DATA_CONFIG": {
            "split_random_seed": 0,
            "dataset": "ph2",
        },
        "TRANSFORM_CONFIG": {
            "seed_value": 0,
            "augment": True,
        },
        "MODEL_CONFIG": {
            "encoder_residual_mode": "classical",
            "merge_residual_mode": "classical",
            "arch": "unet2d",
        },
    }


# ---------------------------------------------------------------------------
# _patch_yaml_dict — seed plumbing
# ---------------------------------------------------------------------------

class TestPatchYamlDictSeeds:
    def test_seed_written_to_all_three_locations(self, base_yaml: dict) -> None:
        result = _patch_yaml_dict(base_yaml, seed=42, experiment_name="exp")
        assert result["TRAIN_CONFIG"]["seed"] == 42
        assert result["DATA_CONFIG"]["split_random_seed"] == 42
        assert result["TRANSFORM_CONFIG"]["seed_value"] == 42

    def test_different_seeds_produce_different_dicts(self, base_yaml: dict) -> None:
        r1 = _patch_yaml_dict(base_yaml, seed=1, experiment_name="exp")
        r2 = _patch_yaml_dict(base_yaml, seed=2, experiment_name="exp")
        assert r1["TRAIN_CONFIG"]["seed"] != r2["TRAIN_CONFIG"]["seed"]

    def test_zero_seed_is_accepted(self, base_yaml: dict) -> None:
        result = _patch_yaml_dict(base_yaml, seed=0, experiment_name="exp")
        assert result["TRAIN_CONFIG"]["seed"] == 0


# ---------------------------------------------------------------------------
# _patch_yaml_dict — experiment_name
# ---------------------------------------------------------------------------

class TestPatchYamlDictExperimentName:
    def test_experiment_name_is_patched(self, base_yaml: dict) -> None:
        result = _patch_yaml_dict(base_yaml, seed=1, experiment_name="my_experiment")
        assert result["TRAIN_CONFIG"]["experiment_name"] == "my_experiment"

    def test_original_experiment_name_unchanged(self, base_yaml: dict) -> None:
        original_name = base_yaml["TRAIN_CONFIG"]["experiment_name"]
        _patch_yaml_dict(base_yaml, seed=1, experiment_name="other")
        assert base_yaml["TRAIN_CONFIG"]["experiment_name"] == original_name


# ---------------------------------------------------------------------------
# _patch_yaml_dict — encoder / merge mode patching
# ---------------------------------------------------------------------------

class TestPatchYamlDictModes:
    def test_encoder_mode_patched_when_provided(self, base_yaml: dict) -> None:
        result = _patch_yaml_dict(base_yaml, seed=1, experiment_name="e", encoder_mode="he2")
        assert result["MODEL_CONFIG"]["encoder_residual_mode"] == "he2"

    def test_merge_mode_patched_when_provided(self, base_yaml: dict) -> None:
        result = _patch_yaml_dict(base_yaml, seed=1, experiment_name="e", merge_mode="attention_gate")
        assert result["MODEL_CONFIG"]["merge_residual_mode"] == "attention_gate"

    def test_encoder_mode_unchanged_when_none(self, base_yaml: dict) -> None:
        original = base_yaml["MODEL_CONFIG"]["encoder_residual_mode"]
        result = _patch_yaml_dict(base_yaml, seed=1, experiment_name="e", encoder_mode=None)
        assert result["MODEL_CONFIG"]["encoder_residual_mode"] == original

    def test_merge_mode_unchanged_when_none(self, base_yaml: dict) -> None:
        original = base_yaml["MODEL_CONFIG"]["merge_residual_mode"]
        result = _patch_yaml_dict(base_yaml, seed=1, experiment_name="e", merge_mode=None)
        assert result["MODEL_CONFIG"]["merge_residual_mode"] == original

    def test_both_modes_patched_together(self, base_yaml: dict) -> None:
        result = _patch_yaml_dict(
            base_yaml, seed=7, experiment_name="e",
            encoder_mode="se", merge_mode="he1",
        )
        assert result["MODEL_CONFIG"]["encoder_residual_mode"] == "se"
        assert result["MODEL_CONFIG"]["merge_residual_mode"] == "he1"


# ---------------------------------------------------------------------------
# _patch_yaml_dict — deep copy isolation
# ---------------------------------------------------------------------------

class TestPatchYamlDictIsolation:
    def test_original_dict_not_mutated(self, base_yaml: dict) -> None:
        original = copy.deepcopy(base_yaml)
        _patch_yaml_dict(base_yaml, seed=99, experiment_name="x",
                         encoder_mode="he2", merge_mode="attention_gate")
        assert base_yaml == original

    def test_two_patches_are_independent(self, base_yaml: dict) -> None:
        r1 = _patch_yaml_dict(base_yaml, seed=1, experiment_name="e1")
        r2 = _patch_yaml_dict(base_yaml, seed=2, experiment_name="e2")
        # Mutating r1 must not affect r2
        r1["MODEL_CONFIG"]["arch"] = "MUTATED"
        assert r2["MODEL_CONFIG"]["arch"] == base_yaml["MODEL_CONFIG"]["arch"]

    def test_non_patched_keys_preserved(self, base_yaml: dict) -> None:
        result = _patch_yaml_dict(base_yaml, seed=1, experiment_name="e")
        assert result["TRAIN_CONFIG"]["lr"] == base_yaml["TRAIN_CONFIG"]["lr"]
        assert result["DATA_CONFIG"]["dataset"] == base_yaml["DATA_CONFIG"]["dataset"]
        assert result["TRANSFORM_CONFIG"]["augment"] == base_yaml["TRANSFORM_CONFIG"]["augment"]
        assert result["MODEL_CONFIG"]["arch"] == base_yaml["MODEL_CONFIG"]["arch"]


# ---------------------------------------------------------------------------
# _config_from_dict — temp-file round-trip (loader mocked)
# ---------------------------------------------------------------------------

class TestConfigFromDict:
    def test_temp_file_is_created_and_loader_receives_yaml_path(self, base_yaml: dict, tmp_path: Path) -> None:
        fake_cfg = MagicMock()
        fake_cfg.cfg_path = ""

        with patch("run_seeds.load_config_from_yaml", return_value=fake_cfg) as mock_loader:
            _config_from_dict(base_yaml, tmp_path / "base.yaml")

        mock_loader.assert_called_once()
        called_path: Path = mock_loader.call_args[0][0]
        assert called_path.exists(), "temp file must still exist when loader is called"
        assert called_path.suffix == ".yaml"

    def test_cfg_path_updated_to_temp_file(self, base_yaml: dict, tmp_path: Path) -> None:
        fake_cfg = MagicMock()
        fake_cfg.cfg_path = ""

        with patch("run_seeds.load_config_from_yaml", return_value=fake_cfg):
            result = _config_from_dict(base_yaml, tmp_path / "base.yaml")

        assert result.cfg_path != str(tmp_path / "base.yaml"), (
            "cfg_path must point at the patched temp file, not the original base config"
        )

    def test_temp_file_contains_patched_values(self, base_yaml: dict, tmp_path: Path) -> None:
        captured_path: list[Path] = []

        def capturing_loader(p: Path) -> MagicMock:
            captured_path.append(p)
            cfg = MagicMock()
            cfg.cfg_path = ""
            return cfg

        with patch("run_seeds.load_config_from_yaml", side_effect=capturing_loader):
            _config_from_dict(base_yaml, tmp_path / "base.yaml")

        written = yaml.safe_load(captured_path[0].read_text())
        assert written["TRAIN_CONFIG"]["seed"] == base_yaml["TRAIN_CONFIG"]["seed"]
        assert written["MODEL_CONFIG"]["arch"] == base_yaml["MODEL_CONFIG"]["arch"]
