"""Unit tests for SkiNet.ML.dataloaders.dataloaders"""

import pytest
from torch.utils.data import SequentialSampler, BatchSampler, RandomSampler
from SkiNet.ML.dataloaders.dataloaders import _RepeatSampler


"""------PARAMS ---------------------------------------------------------------"""
# A simple dataset of 10 elements.
DATASET = list(range(10))


"""------FIXTURES ---------------------------------------------------------------"""
@pytest.fixture
def repeat_sampler_fixture():
    """
    A fixture to create and return a _RepeatSampler based on a SequentialSampler.
    """
    def _create_sampler(batch_size, max_num_to_repeat, drop_last):
        batch_sampler = BatchSampler(
            SequentialSampler(DATASET),
            batch_size=batch_size,
            drop_last=drop_last
        )
        repeat_sampler = _RepeatSampler(
            batch_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            max_num_to_repeat=max_num_to_repeat
        )
        return repeat_sampler
    return _create_sampler

@pytest.fixture
def random_repeat_sampler_fixture():
    """
    A fixture to create and return a _RepeatSampler based on a RandomSampler.
    """
    def _create_sampler(batch_size, max_num_to_repeat, drop_last):
        batch_sampler = BatchSampler(
            RandomSampler(DATASET),
            batch_size=batch_size,
            drop_last=drop_last
        )
        repeat_sampler = _RepeatSampler(
            batch_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            max_num_to_repeat=max_num_to_repeat
        )
        return repeat_sampler
    return _create_sampler

"""------TESTS for _RepeatSampler - SequentialSampler - finite repeat  ---------------------------------------------------------------"""

@pytest.mark.parametrize("batch_size, max_num_to_repeat, drop_last, expected_batches", [
    # Tests with drop_last=False.
    (1, 1, False, [[i] for i in range(10)]),
    (1, 3, False, ([[i] for i in range(10)]) * 3),
    (2, 1, False, [[0,1], [2,3], [4,5], [6,7], [8,9]]),
    (2, 3, False, ([[0,1], [2,3], [4,5], [6,7], [8,9]]) * 3),
    (3, 1, False, [[0,1,2], [3,4,5], [6,7,8], [9]]),
    (3, 2, False, ([[0,1,2], [3,4,5], [6,7,8], [9]]) * 2),
    (4, 1, False, [[0,1,2,3], [4,5,6,7], [8,9]]),
    (4, 2, False, ([[0,1,2,3], [4,5,6,7], [8,9]]) * 2),
    # Tests with drop_last=True.
    (1, 1, True, [[i] for i in range(10)]),
    (1, 3, True, ([[i] for i in range(10)]) * 3),
    (2, 1, True, [[0,1], [2,3], [4,5], [6,7], [8,9]]),
    (2, 3, True, ([[0,1], [2,3], [4,5], [6,7], [8,9]]) * 3),
    (3, 1, True, [[0,1,2], [3,4,5], [6,7,8]]),
    (3, 2, True, ([[0,1,2], [3,4,5], [6,7,8]]) * 2),
    (4, 1, True, [[0,1,2,3], [4,5,6,7]]),
    (4, 2, True, ([[0,1,2,3], [4,5,6,7]]) * 2),
])
def test_repeat_sampler(repeat_sampler_fixture, batch_size, max_num_to_repeat, drop_last, expected_batches):
    """Test that the _RepeatSampler correctly repeats the batches with and without dropping the last batch."""
    # Use the fixture to create the _RepeatSampler.
    repeat_sampler = repeat_sampler_fixture(batch_size, max_num_to_repeat, drop_last)
    # Convert the sampler to a list to get all the sampled batches.
    sampled_batches = list(repeat_sampler)
    # Check if the sampled batches match the expected batches.
    assert sampled_batches == expected_batches, f"Expected {expected_batches}, but got {sampled_batches}"



"""------TESTS for _RepeatSampler - SequentialSampler - infinite repeat ---------------------------------------------------------------"""

@pytest.mark.parametrize("batch_size, drop_last, expected_batches", [
    # Tests with drop_last=False.
    (1, False, [[i] for i in range(10)]),
    (2, False, [[0,1], [2,3], [4,5], [6,7], [8,9]]),
    (3, False, [[0,1,2], [3,4,5], [6,7,8], [9]]),
    (4, False, [[0,1,2,3], [4,5,6,7], [8,9]]),
    # Tests with drop_last=True.
    (1, True, [[i] for i in range(10)]),
    (2, True, [[0,1], [2,3], [4,5], [6,7], [8,9]]),
    (3, True, [[0,1,2], [3,4,5], [6,7,8]]),
    (4, True, [[0,1,2,3], [4,5,6,7]]),
])
def test_infinite_repeat_sampler(repeat_sampler_fixture, batch_size, drop_last, expected_batches):
    """
    Test that the _RepeatSampler continuously repeats the batches infinitely.
    """
    # Use the fixture to create the infinite _RepeatSampler.
    repeat_sampler = repeat_sampler_fixture(batch_size, max_num_to_repeat=0, drop_last=drop_last)

    # Convert the sampler to a list of sampled batches but limit the iterations to avoid an infinite loop.
    NUM_EPOCHS = 10  # Number of times the dataset should repeat, i.e. NUM_EPOCHS
    repeat_sampler_batches = []
    for i, batch in enumerate(repeat_sampler):
        if i >= NUM_EPOCHS * len(expected_batches):  # Stop after the dataset has repeated NUM_EPOCHS times
            break
        repeat_sampler_batches.append(batch)

    # Generate the expected batches repeated NUM_EPOCHS times.
    expected_repeated_batches = expected_batches * NUM_EPOCHS

    # Check if the sampled batches match the expected repeated batches.
    assert repeat_sampler_batches == expected_repeated_batches, f"Expected {expected_repeated_batches}, but got {repeat_sampler_batches}"


"""------TESTS for _RepeatSampler - RandomSampler - finite repeat ---------------------------------------------------------------"""

@pytest.mark.parametrize("batch_size, max_num_to_repeat, drop_last, num_batches", [
    # Tests with drop_last=False.
    (2, 1, False, 5),  # 10 elements, batch size 2 => 5 batches
    (2, 3, False, 15),  # 10 elements, batch size 2 => 5 batches; 3 repeats * 5 batches = 15 batches
    (3, 1, False, 4),  # 10 elements, batch size 3 => 4 batches (last is incomplete)
    (3, 2, False, 8),  # 10 elements, batch size 3 => 4 batches; 2 repeats * 4 batches = 8 batches
    # Tests with drop_last=True.
    (3, 1, True, 3),  # 10 elements, batch size 3 => 3 full batches
    (3, 2, True, 6),  # 10 elements, batch size 3 => 3 full batches; 2 repeats * 3 full batches = 6 batches
])
def test_random_repeat_sampler(random_repeat_sampler_fixture, batch_size, max_num_to_repeat, drop_last, num_batches):
    """Test that the _RepeatSampler with RandomSampler generates the correct number of batches."""
    repeat_sampler = random_repeat_sampler_fixture(batch_size, max_num_to_repeat, drop_last)
    repeat_sampler_batches = list(repeat_sampler)

    # Check if the number of batches matches the expectation.
    assert len(repeat_sampler_batches) == num_batches, f"Expected {num_batches} batches, but got {len(repeat_sampler_batches)}"

    # Verify that the number of items yielded by repeat sampler is correct
    # Get the number of items that is expected to be delivered by the repeat sampler
    if drop_last:
        num_expected_total_elements = max_num_to_repeat * batch_size * (len(DATASET) // batch_size)  # Only full batches
    else:
        num_expected_total_elements = max_num_to_repeat * len(DATASET)

    # Get the actual items delivered 
    repeat_sampler_items = [item for batch in repeat_sampler_batches for item in batch]

    # Compare the length of the items delivered with the expected number of items
    assert len(repeat_sampler_items) == num_expected_total_elements, (
        f"Expected {num_expected_total_elements} elements, but got {len(repeat_sampler_items)}"
    )
    assert set(repeat_sampler_items).issubset(set(DATASET)), (
        f"Expected elements from {DATASET} to appear in random order, but got {repeat_sampler_items}"
    )

"""------TESTS for _RepeatSampler - RandomSampler - infinite repeat ---------------------------------------------------------------"""

@pytest.mark.parametrize("batch_size, drop_last, num_epochs", [
    # Tests with drop_last=False.
    (2, False, 3),  # 10 elements, batch size 2 => 5 batches; 3 epochs = 15 batches
    (3, False, 2),  # 10 elements, batch size 3 => 4 batches; 2 epochs = 8 batches
    # Tests with drop_last=True.
    (3, True, 3),  # 10 elements, batch size 3 => 3 full batches; 3 epochs = 9 batches
    (4, True, 2),  # 10 elements, batch size 4 => 2 full batches; 2 epochs = 4 batches
])
def test_random_infinite_repeat_sampler(random_repeat_sampler_fixture, batch_size, drop_last, num_epochs):
    """
    Test that the _RepeatSampler with RandomSampler infinitely repeats the dataset.
    """
    # Use the fixture to create the infinite _RepeatSampler.
    repeat_sampler = random_repeat_sampler_fixture(batch_size, max_num_to_repeat=0, drop_last=drop_last)

    # Limit the iterations to simulate a finite number of "epochs" for testing purposes
    repeat_sampler_batches = []
    num_batches_per_epoch = (len(DATASET) // batch_size) if drop_last else (len(DATASET) + batch_size - 1) // batch_size
    max_batches = num_epochs * num_batches_per_epoch

    for i, batch in enumerate(repeat_sampler):
        if i >= max_batches:
            break
        repeat_sampler_batches.append(batch)

    # Check if the number of batches matches the expectation
    assert len(repeat_sampler_batches) == max_batches, f"Expected {max_batches} batches, but got {len(repeat_sampler_batches)}"

    # Verify that the number of items yielded by the sampler matches the expected total
    if drop_last:
        num_expected_total_elements = num_epochs * batch_size * (len(DATASET) // batch_size)  # Only full batches
    else:
        num_expected_total_elements = num_epochs * len(DATASET)

    repeat_sampler_items = [item for batch in repeat_sampler_batches for item in batch]
    assert len(repeat_sampler_items) == num_expected_total_elements, (
        f"Expected {num_expected_total_elements} elements, but got {len(repeat_sampler_items)}"
    )
    assert set(repeat_sampler_items).issubset(set(DATASET)), (
        f"Expected elements from {DATASET} to appear in random order, but got {repeat_sampler_items}"
    )