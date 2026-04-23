from pathlib import Path
import random
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import logging

from SkiNet.ML.datasets.sample_specs import Sample
logger = logging.getLogger(__name__)


def visualize_augmented_data(dataset: Any,
                             samples: int = 2,
                             save_dir: str | Path | None = "./augmented_data_vis",
                             prefix: str = "vis",
                             show: bool = False) -> None:
    """
    Visualize augmented samples without affecting the global RNG state.
    RNG is snapshot before and restored after to ensure training reproducibility
    is not affected by visualization.
    """
    torch_state = torch.get_rng_state()
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

    try:
        _do_visualize(dataset=dataset,
                      samples=samples,
                      save_dir=save_dir,
                      prefix=prefix,
                      show=show)
    finally:
        torch.set_rng_state(torch_state)
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state)


def _do_visualize(dataset: Any,
                  samples: int,
                  save_dir: str | Path | None,
                  prefix: str,
                  show: bool) -> None:
    """
    Visualize `samples` randomly chosen dataset entries. Each row shows:
    original image | augmented image | augmented mask (with non-binary pixel check).

    :param dataset: Dataset to visualize from. Expected to have get_raw_sample,
        sample_ids, sample_specs, and transform attributes.
    :param samples: Number of randomly chosen samples to visualize.
    :param save_dir: Directory to save visualizations. No saving if None.
    :param prefix: Prefix for saved file names.
    :param show: Whether to show the figure interactively.
    """
    save_path = Path(save_dir) if save_dir is not None else None
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)

    indices = random.sample(range(len(dataset)), k=min(samples, len(dataset)))
    vis_transform = _get_visualization_transform(dataset.transform)

    figure, axes = plt.subplots(len(indices), 3, figsize=(12, 4 * len(indices)))
    # ensure axes is always 2D even for a single sample
    if len(indices) == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        sample_id = dataset.sample_ids[idx] if hasattr(dataset, "sample_ids") else str(idx)
        base = f"{prefix}_{sample_id}_idx{idx}"

        raw = dataset.get_raw_sample(idx)
        image_t = raw.image
        mask_t = raw.mask

        image = _to_numpy_hwc_for_plot(image_t)
        mask = _to_numpy_hw_for_plot(mask_t)

        # apply visualization transform (without postprocessing)
        specs_item = dataset.sample_specs[dataset.sample_ids[idx]]
        sample = Sample(image=image_t, mask=mask_t, specs=specs_item)
        transformed = vis_transform(sample) if vis_transform is not None else sample
        aug_image = _to_numpy_hwc_for_plot(transformed.image)
        aug_mask = _to_numpy_hw_for_plot(transformed.mask)
        aug_mask_not_binary = (~np.isin(aug_mask, [0, 255])).astype(np.uint8)

        # col 0: original overlay
        axes[row, 0].imshow(overlay_mask(_img_for_overlay(image), mask))
        axes[row, 0].set_title(f"Original (id={sample_id})")
        axes[row, 0].axis("off")

        # col 1: augmented overlay
        axes[row, 1].imshow(overlay_mask(_img_for_overlay(aug_image), aug_mask))
        axes[row, 1].set_title(f"Augmented (id={sample_id})")
        axes[row, 1].axis("off")

        # col 2: non-binary mask pixels (interpolation artefact check)
        axes[row, 2].imshow(aug_mask_not_binary, cmap="gray")
        axes[row, 2].set_title("Non-binary mask pixels")
        axes[row, 2].axis("off")

        if save_path is not None:
            plt.imsave(save_path / f"{base}_orig_overlay.png",
                       overlay_mask(_img_for_overlay(image), mask))
            plt.imsave(save_path / f"{base}_aug_overlay.png",
                       overlay_mask(_img_for_overlay(aug_image), aug_mask))
            plt.imsave(save_path / f"{base}_aug_mask_not_binary.png",
                       aug_mask_not_binary, cmap="gray")

    plt.tight_layout()
    if save_path is not None:
        figure.savefig(save_path / f"{prefix}_grid.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)


def overlay_mask(image: npt.NDArray[Any],
                 mask: npt.NDArray[Any],
                 alpha: float = 0.5,
                 color: tuple = (0, 1, 0)) -> npt.NDArray[Any]:
    mask_overlay = np.zeros_like(image, dtype=np.uint8)
    mask_overlay[mask > 0] = (np.array(color) * 255).astype(np.uint8)
    return np.asarray(cv2.addWeighted(image, 1, mask_overlay, alpha, 0))


def _to_numpy_hwc_for_plot(img: object) -> npt.NDArray[Any]:
    if isinstance(img, torch.Tensor):
        x = img.detach().cpu()
        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = x.permute(1, 2, 0)
        return np.asarray(x.numpy())
    if isinstance(img, np.ndarray):
        return np.asarray(img)
    raise TypeError(f"Unsupported image type for plotting: {type(img)}")


def _to_numpy_hw_for_plot(mask: object) -> npt.NDArray[Any]:
    if isinstance(mask, torch.Tensor):
        tensor = mask.detach().cpu()
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim == 3 and tensor.shape[-1] == 1:
            tensor = tensor[..., 0]
        return np.asarray(tensor.numpy())
    if isinstance(mask, np.ndarray):
        arr = np.asarray(mask)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.asarray(arr[..., 0])
        return arr
    raise TypeError(f"Unsupported mask type for plotting: {type(mask)}")


def _img_for_imsave(img: npt.NDArray[Any]) -> npt.NDArray[Any]:
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        mn = float(np.nanmin(arr)) if arr.size else 0.0
        mx = float(np.nanmax(arr)) if arr.size else 1.0
        if mn >= 0.0 and mx <= 1.0:
            return np.asarray(arr, dtype=np.float32)
        if mx > mn:
            arr = np.asarray((arr - mn) / (mx - mn), dtype=np.float32)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)
        return np.asarray(np.clip(arr, 0.0, 1.0), dtype=np.float32)
    return np.asarray(np.clip(arr, 0, 255), dtype=np.uint8)


def _img_for_overlay(img: npt.NDArray[Any]) -> npt.NDArray[Any]:
    arr = _img_for_imsave(img)
    if arr.dtype == np.uint8:
        return arr
    return np.asarray((arr * 255.0).round().clip(0, 255), dtype=np.uint8)


def _get_visualization_transform(dataset_transform: Any) -> Any | None:
    if dataset_transform is None:
        return None
    if hasattr(dataset_transform, "without_postprocess"):
        return dataset_transform.without_postprocess()
    logger.warning("Transform does not implement without_postprocess(); "
                   "visualization will include postprocessing steps.")
    return dataset_transform
