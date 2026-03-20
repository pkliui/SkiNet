from SkiNet.ML.datasets.sample_specs import Sample


def crop_2d_image(sample: Sample, crop_size: tuple[int, int]) -> list[slice]:
    """
    Crops a 2D image around a specified center.

    :param crop_size: (crop_height, crop_width).
    :return: Cropped image.
    """
    sample_size = sample.image.shape
    center = (crop_size[0] // 2, crop_size[1] // 2)

    slices = []
    for i in range(len(crop_size)):
        if crop_size[i] > sample_size[i]:
            raise ValueError(f"Crop size {crop_size[i]} exceeds image size {sample_size[i]} in dimension {i}")
        start = max(center[i] - crop_size[i] // 2, 0)
        end = min(start + crop_size[i], sample_size[i])
        # Adjust start if end-start < crop_size (near boundary)
        start = max(end - crop_size[i], 0)
        slices.append(slice(start, end))
    return slices
