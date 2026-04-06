from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch

from SkiNet.ML.datasets.sample_specs import Sample


def visualize_augmented_data(dataset: Any,
                             idx: int = 0,
                             samples: int = 2,
                             save_dir: str | Path | None = "./augmented_data_vis",
                             prefix: str = "vis",
                             save_overlay: bool = True,
                             show: bool = False) -> None:
    """
    Visualize one raw sample at index idx and its multiple transformed variants from the dataset.

    :param dataset: The dataset to visualize from, expected to have a
        get_raw_sample method and sample_specs attribute.
    :param idx: The index of the sample to visualize.
    :param samples: The number of augmented samples to visualize.
    :param save_dir: Directory to save the visualizations. If None,
        visualizations will not be saved.
    :param prefix: Prefix for saved file names.
    :param save_overlay: Whether to save overlay images with masks.
    :param show: Whether to show the visualizations.
    """

    save_path = Path(save_dir) if save_dir is not None else None
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)

    sample_id = dataset.sample_ids[idx] if hasattr(dataset, "sample_ids") else str(idx)
    base = f"{prefix}_{sample_id}_idx{idx}"

    figure, ax = plt.subplots(samples + 1, 2, figsize=(8, 4 * (samples + 1)))

    raw = dataset.get_raw_sample(idx)
    image_t = raw.image
    mask_t = raw.mask

    image = _to_numpy_hwc_for_plot(image_t)
    mask = _to_numpy_hw_for_plot(mask_t)

    ax[0, 0].imshow(image)
    ax[0, 0].set_title("Original Image")
    ax[0, 0].axis("off")

    ax[0, 1].imshow(mask, cmap="gray")
    ax[0, 1].set_title("Original Mask")
    ax[0, 1].axis("off")

    if save_path is not None:
        plt.imsave(save_path / f"{base}_orig_image.png", _img_for_imsave(image))
        plt.imsave(save_path / f"{base}_orig_mask.png", mask, cmap="gray")
        if save_overlay:
            plt.imsave(save_path / f"{base}_orig_overlay.png",
                       overlay_mask(_img_for_overlay(image), mask))

    vis_transform = _get_visualization_transform(dataset.transform)

    for i in range(samples):
        specs_item = dataset.sample_specs[dataset.sample_ids[idx]]
        sample = Sample(image=image_t, mask=mask_t, specs=specs_item)
        transformed = vis_transform(sample) if vis_transform is not None else sample

        aug_image = _to_numpy_hwc_for_plot(transformed.image)
        aug_mask = _to_numpy_hw_for_plot(transformed.mask)

        ax[i + 1, 0].imshow(_img_for_overlay(aug_image))
        ax[i + 1, 0].set_title(f"Augmented Image {i + 1}")
        ax[i + 1, 0].axis("off")

        ax[i + 1, 1].imshow(aug_mask, cmap="gray")
        ax[i + 1, 1].set_title(f"Augmented Mask {i + 1}")
        ax[i + 1, 1].axis("off")

        if save_path is not None:
            plt.imsave(save_path / f"{base}_aug{i + 1}_image.png", _img_for_imsave(aug_image))
            plt.imsave(save_path / f"{base}_aug{i + 1}_mask.png", aug_mask, cmap="gray")
            if save_overlay:
                plt.imsave(save_path / f"{base}_aug{i + 1}_overlay.png",
                           overlay_mask(_img_for_overlay(aug_image), aug_mask))

    plt.tight_layout()
    if save_path is not None:
        figure.savefig(save_path / f"{base}_grid.png", dpi=150, bbox_inches="tight")

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
    return dataset_transform
