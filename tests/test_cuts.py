import tempfile
from fact.io import to_h5py
import pandas as pd
import h5py


df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [-2, 3, 5, 1]})

def test_multiple_config():
    from aict_tools.apply import create_mask_h5py

    config = [{'b': ['>', 0]}, {'b': ['<', 5]}]

    with tempfile.NamedTemporaryFile(prefix='test_aict_', suffix='.hdf5') as f:
        to_h5py(df, f.name, key='events')

        mask = create_mask_h5py(
            h5py.File(f.name, 'r'), n_events=len(df), selection_config=config
        )
        assert all(mask == [False, True, False, True])
