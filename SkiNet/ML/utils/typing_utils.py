from typing import Tuple, Union

IntOrTuple2d3d = int | Tuple[int, int] | Tuple[int, int, int]
TupleOfInt2d3d = Tuple[int, int] | Tuple[int, int, int]


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
