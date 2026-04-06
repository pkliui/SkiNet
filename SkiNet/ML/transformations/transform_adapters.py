from dataclasses import dataclass
from typing import Protocol

import albumentations as A
import torch

from SkiNet.ML.datasets.sample_specs import Sample
from SkiNet.ML.transformations.transform_utils import convert_to_hwc_numpy


class SampleTransformAdapter(Protocol):
    """
    Protocol for transforming a Sample object.
    """

    def __call__(self, sample: Sample) -> Sample:
        ...


@dataclass(frozen=True)
class AlbumentationsSampleTransform:
    """
    Wrap a single Albumentations graph behind the Sample -> Sample interface.

    The main pipeline is expected to include postprocessing and therefore return
    torch tensors. The visualization pipeline omits postprocessing and returns
    numpy arrays.

    """

    pipeline: A.Compose
    visualization_pipeline: A.Compose | None = None
    expects_tensor_output: bool = True

    def __call__(self, sample: Sample) -> Sample:
        """
        Returns a transformed Sample by applying the Albumentations pipeline
        to the input Sample's image and mask.

        :param sample: A Sample object containing the image and mask to be transformed.
            Image and mask are in CHW, uint8 format as loaded from disk and are converted to HWC numpy arrays for compatibility with Albumentations.
        :return: A new Sample object with the transformed image and mask.
        The output format depends on the final transforms in the Albumentations pipeline, and as long as ToTensorV2(transpose_mask=True) is included,
         the output will be torch tensors CHW for both image (float32) and mask (uint8).
        """
        out = self.pipeline(image=convert_to_hwc_numpy(sample.image),
                            mask=convert_to_hwc_numpy(sample.mask))
        out_sample = sample.model_copy(update={"image": out["image"], "mask": out["mask"]})

        if self.expects_tensor_output:
            if not isinstance(out_sample.image, torch.Tensor) or not isinstance(out_sample.mask, torch.Tensor):
                raise TypeError("Transform pipeline must output torch tensors. "
                                "Ensure the Albumentations graph ends with ToTensorV2(transpose_mask=True).")

        return out_sample

    def without_postprocess(self) -> "AlbumentationsSampleTransform":
        vis_pipeline = self.visualization_pipeline
        if vis_pipeline is None:
            vis_pipeline = self.pipeline
        return AlbumentationsSampleTransform(pipeline=vis_pipeline,
                                             visualization_pipeline=vis_pipeline,
                                             expects_tensor_output=False)
