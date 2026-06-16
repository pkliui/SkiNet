"""
Export a Lightning checkpoint → skinet_unet.onnx

Usage:
    cd repos/SkiNet
    # Provide just the MLflow run folder — checkpoint and config are auto-discovered:
    python export_onnx.py --run mlruns/E4-isic2017-unet2d-thres-sweep-part2/1/a8cb781b18f64549a0d36ca9e096cee5 --out ios_onnx.onnx

    # Or supply paths explicitly:
    python export_onnx.py --ckpt path/to/epoch.ckpt --config path/to/config.yaml --out ios_onnx.onnx --opset 17

"""

from SkiNet.ML.model.lightning_model import build_lightning_model
from SkiNet.ML.configs.load_config_from_yaml import load_config_from_yaml
from pathlib import Path
import torch.nn as nn
import torch
import argparse
import os
# Both fixes are needed: LOGNAME satisfies getpass.getuser() (called from multiple torch.compile
# internals); TORCHINDUCTOR_CACHE_DIR covers the inductor path that fires before getuser().
os.environ.setdefault("LOGNAME", "user")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor_cache")


INPUT_SIZE = 256


class _UNetWithSigmoid(nn.Module):
    """Wraps the UNet backbone so the ONNX graph outputs probabilities, not logits."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.backbone(x))


def _unwrap_compiled(model: nn.Module) -> nn.Module:
    """Strip torch.compile wrapper if present (training used use_torch_compile=True)."""
    return getattr(model, "_orig_mod", model)


def export(ckpt_path: Path, out_path: Path, config_path: Path, opset: int = 17) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading config from {config_path} …")
    config = load_config_from_yaml(config_path)

    print("Building model architecture …")
    config.trainconfig.use_torch_compile = False  # not needed for export, avoids env issues
    lightning_model = build_lightning_model(config)

    print(f"Loading checkpoint: {ckpt_path} …")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Checkpoint was saved with torch.compile active, so keys are prefixed "model._orig_mod.*".
    # Strip the extra prefix so they match the uncompiled model we built for export.
    state_dict = {k.replace("model._orig_mod.", "model."): v for k, v in ckpt["state_dict"].items()}
    lightning_model.load_state_dict(state_dict)
    lightning_model.eval()

    # Report the optimal threshold stored in the checkpoint so the iOS app can use it
    threshold = lightning_model.optimal_threshold.item()
    print(f"Optimal sigmoid threshold from checkpoint: {threshold:.4f}")
    print("  → Hard-code this value as SEGMENTATION_THRESHOLD in your iOS app's ml/modelRunner.ts")

    backbone = _unwrap_compiled(lightning_model.model)
    export_model = _UNetWithSigmoid(backbone)
    export_model.eval()

    dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)

    print(f"Exporting to ONNX (opset {opset}) → {out_path} …")
    torch.onnx.export(
        export_model,
        (dummy,),
        str(out_path),
        opset_version=opset,
        input_names=["image"],
        output_names=["mask_prob"],
        dynamic_axes={"image": {0: "batch"}, "mask_prob": {0: "batch"}},
        do_constant_folding=True,
    )
    print("Export complete.")

    # The dynamo exporter may write weights as a separate .data sidecar file.
    # Merge everything into a single self-contained file for iOS deployment.
    try:
        import onnx
        model_proto = onnx.load(str(out_path))
        onnx.save_model(
            model_proto,
            str(out_path),
            save_as_external_data=False,
        )
        # Remove the stale sidecar if it still exists after merging
        sidecar = out_path.with_suffix(out_path.suffix + ".data")
        if sidecar.exists():
            sidecar.unlink()
        print("Weights merged into single self-contained ONNX file.")
    except ImportError:
        print("onnx not installed — skipping weight merge (pip install onnx to enable)")

    # Quick validation with onnxruntime if available
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        dummy_np = np.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
        output = sess.run(["mask_prob"], {"image": dummy_np})[0]
        assert output.shape == (1, 1, INPUT_SIZE, INPUT_SIZE), f"Unexpected output shape: {output.shape}"
        print(f"ONNXRuntime validation passed — output shape: {output.shape}")
    except ImportError:
        print("onnxruntime not installed — skipping validation (pip install onnxruntime to enable)")

    size_bytes = out_path.stat().st_size
    size_mb = size_bytes / 1024 / 1024
    size_str = f"{size_mb:.1f} MB" if size_mb >= 1 else f"{size_bytes / 1024:.0f} KB"
    print(f"\nModel size: {size_str}")
    print("\nPreprocessing constants for your iOS app:")
    print(f"  INPUT_SIZE  = {INPUT_SIZE}")
    print("  NORM_MEAN   = [0.699, 0.556, 0.5121]")
    print("  NORM_STD    = [0.1576, 0.1562, 0.1706]")
    print(f"  THRESHOLD   = {threshold:.4f}")


def _resolve_run(run_dir: Path) -> tuple[Path, Path]:
    """Auto-discover checkpoint and config inside an MLflow run folder."""
    ckpts = sorted(run_dir.glob("artifacts/checkpoints/**/*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt found under {run_dir}/artifacts/checkpoints/")
    # Prefer 'best/' subfolder if present, otherwise take the last by name
    best = [p for p in ckpts if "best" in p.parts]
    ckpt_path = best[-1] if best else ckpts[-1]

    configs = sorted(run_dir.glob("artifacts/config/*.yaml"))
    if not configs:
        raise FileNotFoundError(f"No .yaml found under {run_dir}/artifacts/config/")
    config_path = configs[-1]

    return ckpt_path, config_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=None,
                        help="MLflow run folder; checkpoint and config are auto-discovered")
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("skinet_unet.onnx"))
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    if args.run is not None:
        ckpt, cfg = _resolve_run(args.run)
        print(f"Auto-discovered checkpoint : {ckpt}")
        print(f"Auto-discovered config     : {cfg}")
    else:
        if args.ckpt is None or args.config is None:
            parser.error("Provide --run <run_dir> or both --ckpt and --config")
        ckpt, cfg = args.ckpt, args.config

    export(ckpt, args.out, cfg, args.opset)
