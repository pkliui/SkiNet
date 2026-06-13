"""
Unit tests for export_onnx.py.

All tests are offline — no real checkpoint, no ONNX runtime required.
They cover the three pure-logic units:
  - _UNetWithSigmoid  (sigmoid wrapper)
  - _unwrap_compiled  (torch.compile unwrapping)
  - _resolve_run      (MLflow run folder discovery)
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path

from export_onnx import _UNetWithSigmoid, _unwrap_compiled, _resolve_run


# ---------------------------------------------------------------------------
# _UNetWithSigmoid
# ---------------------------------------------------------------------------

class _ConstantLogitNet(nn.Module):
    """Returns a fixed logit tensor so we can verify the sigmoid is applied."""

    def __init__(self, logit: float) -> None:
        super().__init__()
        self.logit = logit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, self.logit)


def test_unet_with_sigmoid_applies_sigmoid_to_large_positive_logit() -> None:
    backbone = _ConstantLogitNet(100.0)
    model = _UNetWithSigmoid(backbone)
    x = torch.zeros(1, 3, 4, 4)
    out = model(x)
    assert torch.allclose(out, torch.ones_like(out), atol=1e-5), \
        "sigmoid(+100) should be ~1.0"


def test_unet_with_sigmoid_applies_sigmoid_to_large_negative_logit() -> None:
    backbone = _ConstantLogitNet(-100.0)
    model = _UNetWithSigmoid(backbone)
    x = torch.zeros(1, 3, 4, 4)
    out = model(x)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-5), \
        "sigmoid(-100) should be ~0.0"


def test_unet_with_sigmoid_zero_logit_gives_half() -> None:
    backbone = _ConstantLogitNet(0.0)
    model = _UNetWithSigmoid(backbone)
    x = torch.zeros(1, 3, 4, 4)
    out = model(x)
    assert torch.allclose(out, torch.full_like(out, 0.5), atol=1e-6)


def test_unet_with_sigmoid_output_shape_matches_input() -> None:
    backbone = nn.Identity()
    model = _UNetWithSigmoid(backbone)
    x = torch.randn(2, 1, 8, 8)
    out = model(x)
    assert out.shape == x.shape


def test_unet_with_sigmoid_output_range() -> None:
    backbone = nn.Linear(4, 4, bias=False)
    model = _UNetWithSigmoid(backbone)
    x = torch.randn(8, 4) * 10
    out = model(x)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_unet_with_sigmoid_stores_backbone() -> None:
    backbone = nn.Conv2d(3, 1, 1)
    model = _UNetWithSigmoid(backbone)
    assert model.backbone is backbone


# ---------------------------------------------------------------------------
# _unwrap_compiled
# ---------------------------------------------------------------------------

def test_unwrap_compiled_passthrough_when_no_orig_mod() -> None:
    m = nn.Linear(2, 2)
    assert _unwrap_compiled(m) is m


def test_unwrap_compiled_returns_orig_mod_attribute() -> None:
    inner = nn.Linear(2, 2)
    wrapper = nn.Module()
    wrapper._orig_mod = inner
    assert _unwrap_compiled(wrapper) is inner


def test_unwrap_compiled_does_not_recurse_past_one_level() -> None:
    """_unwrap_compiled strips one compile layer only — that is the contract."""
    innermost = nn.Linear(2, 2)
    middle = nn.Module()
    middle._orig_mod = innermost
    outer = nn.Module()
    outer._orig_mod = middle
    # one call should land on `middle`, not `innermost`
    assert _unwrap_compiled(outer) is middle


# ---------------------------------------------------------------------------
# _resolve_run
# ---------------------------------------------------------------------------

def _make_run_dir(tmp_path: Path, ckpt_names: list[str], config_names: list[str]) -> Path:
    """Populate a fake MLflow run folder with the given checkpoint and config files."""
    run_dir = tmp_path / "run"
    ckpt_dir = run_dir / "artifacts" / "checkpoints"
    cfg_dir = run_dir / "artifacts" / "config"
    ckpt_dir.mkdir(parents=True)
    cfg_dir.mkdir(parents=True)
    for name in ckpt_names:
        (ckpt_dir / name).write_text("")
    for name in config_names:
        (cfg_dir / name).write_text("")
    return run_dir


def test_resolve_run_picks_only_checkpoint(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path, ["epoch=5.ckpt"], ["config.yaml"])
    ckpt, cfg = _resolve_run(run_dir)
    assert ckpt.name == "epoch=5.ckpt"
    assert cfg.name == "config.yaml"


def test_resolve_run_prefers_best_subfolder(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    best_dir = run_dir / "artifacts" / "checkpoints" / "best"
    best_dir.mkdir(parents=True)
    other_dir = run_dir / "artifacts" / "checkpoints" / "last"
    other_dir.mkdir(parents=True)
    cfg_dir = run_dir / "artifacts" / "config"
    cfg_dir.mkdir(parents=True)

    (other_dir / "epoch=10.ckpt").write_text("")
    (best_dir / "epoch=8.ckpt").write_text("")
    (cfg_dir / "config.yaml").write_text("")

    ckpt, _ = _resolve_run(run_dir)
    assert "best" in ckpt.parts


def test_resolve_run_falls_back_to_last_by_name_when_no_best(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path, ["epoch=1.ckpt", "epoch=9.ckpt", "epoch=3.ckpt"], ["config.yaml"])
    ckpt, _ = _resolve_run(run_dir)
    # sorted() → lexicographic, so "epoch=9.ckpt" is last
    assert ckpt.name == "epoch=9.ckpt"


def test_resolve_run_raises_when_no_checkpoint(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path, [], ["config.yaml"])
    with pytest.raises(FileNotFoundError, match=r"No \.ckpt found"):
        _resolve_run(run_dir)


def test_resolve_run_raises_when_no_config(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path, ["epoch=0.ckpt"], [])
    with pytest.raises(FileNotFoundError, match=r"No \.yaml found"):
        _resolve_run(run_dir)


def test_resolve_run_picks_last_config_when_multiple(tmp_path: Path) -> None:
    run_dir = _make_run_dir(tmp_path, ["epoch=0.ckpt"], ["a_config.yaml", "z_config.yaml"])
    _, cfg = _resolve_run(run_dir)
    assert cfg.name == "z_config.yaml"


def test_resolve_run_checkpoint_in_nested_subfolder(tmp_path: Path) -> None:
    """glob('**/*.ckpt') must find checkpoints more than one level deep."""
    run_dir = tmp_path / "run"
    deep_dir = run_dir / "artifacts" / "checkpoints" / "best" / "v1"
    deep_dir.mkdir(parents=True)
    cfg_dir = run_dir / "artifacts" / "config"
    cfg_dir.mkdir(parents=True)
    (deep_dir / "model.ckpt").write_text("")
    (cfg_dir / "config.yaml").write_text("")

    ckpt, _ = _resolve_run(run_dir)
    assert ckpt.name == "model.ckpt"
