import numpy as np
from aict_tools.scripts import split_data
import pandas as pd


def test_array_splitting():

    test_df = pd.DataFrame(
        {"array_event_id": np.repeat(np.arange(10), 2)}, index=np.arange(20)
    )

    num_ids = split_data.split_indices(test_df.index, n_total=20, fractions=[0.5, 0.5])
    assert len(num_ids) == 2
    assert num_ids == [10, 10]

    num_ids = split_data.split_indices(
        test_df.array_event_id, n_total=10, fractions=[0.5, 0.5]
    )
    assert len(num_ids) == 2
    assert num_ids == [5, 5]


def test_undividable_length():
    a = np.arange(17)

    num_ids = split_data.split_indices(a, n_total=len(a), fractions=[0.5, 0.5])

    assert len(num_ids) == 2
    assert num_ids == [9, 8]


def test_unequal_fractions():
    a = np.arange(100)

    num_ids = split_data.split_indices(a, n_total=len(a), fractions=[0.4, 0.4, 0.2])

    assert len(num_ids) == 3
    assert num_ids == [40, 40, 20]


def test_unequal_undividable_fractions():
    a = np.arange(101)

    num_ids = split_data.split_indices(a, n_total=len(a), fractions=[0.4, 0.4, 0.2])
    assert len(num_ids) == 3
    assert num_ids == [41, 41, 19]

    a = np.arange(99)

    num_ids = split_data.split_indices(a, n_total=len(a), fractions=[0.4, 0.4, 0.2])
    assert len(num_ids) == 3
    assert num_ids == [40, 40, 19]

    a = np.arange(93)
    num_ids = split_data.split_indices(a, n_total=len(a), fractions=[0.4, 0.4, 0.2])
    assert len(num_ids) == 3
    assert num_ids == [38, 38, 17]
