import pytest

from SkiNet.ML.utils.typing_utils import expand_to_tuple


@pytest.mark.parametrize(
    "value,size,expected",
    [
        (1, 2, (1, 1)),
        (3, 3, (3, 3, 3)),
    ],
)
def test_expand_to_tuple_valid(value: int, size: int, expected: tuple) -> None:
    """
    Test expand_to_tuple expands the input as required
    """
    result = expand_to_tuple(value, size)
    assert isinstance(result, tuple)
    assert result == expected


@pytest.mark.parametrize("size", [0, 1, 4, -1])
def test_expand_to_tuple_invalid_size(size: int) -> None:
    """
    Test an error is raised if the size is not expected
    """
    with pytest.raises(ValueError, match="Invalid size"):
        expand_to_tuple(3, size)
