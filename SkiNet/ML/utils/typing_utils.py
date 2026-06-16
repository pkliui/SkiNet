from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, TypeVar

from torch.utils.data import Dataset

if TYPE_CHECKING:
    from SkiNet.ML.datasets.segmentation_dataset import BaseDataset as DatasetBound
else:
    DatasetBound = Dataset

IntOrTuple2d = int | Tuple[int, int]
TupleOfInt2d = Tuple[int, int]
IntOrTuple3d = int | Tuple[int, int, int]
TupleOfInt3d = Tuple[int, int, int]
IntOrTuple2d3d = int | Tuple[int, int] | Tuple[int, int, int]
TupleOfInt2d3d = Tuple[int, int] | Tuple[int, int, int]

# use covariance to allow DatasetFactory[SegmentationDataset] to be treated as a subtype of DatasetFactory[BaseDataset]
# DatasetFactory is read-only with respect to the dataset type it produces, so covariance is appropriate here
TDataset_co = TypeVar("TDataset_co", bound=DatasetBound, covariant=True)


def expand_to_tuple(value: int, size: int) -> TupleOfInt2d3d:
    """
    Expand a scalar integer into a 2D or 3D tuple.

    :param value: integer to expand
    :param size: size of the target tuple

    return: a tuple of length `size` where all elements equal `value`.

    """
    if size == 2:
        return (value, value)
    elif size == 3:
        return (value, value, value)
    raise ValueError("Invalid size, only 2 or 3 supported")
