import tempfile
import os
import shutil
import numpy as np
import pytest


@pytest.fixture(scope='function')
def tables_file(request, tmpdir_factory):
    fn = tmpdir_factory.mktemp('aict_test_data').join('test_file.hdf5')
    shutil.copy('examples/cta_tables_file.hdf5', fn)
    return fn

@pytest.fixture(scope='function')
def h5py_file(tmpdir_factory):
    fn = tmpdir_factory.mktemp('aict_test_data').join('test_file.hdf5')
    shutil.copy('examples/gamma.hdf5', fn)
    return fn

@pytest.fixture(scope='session')
def fact_config():
    from aict_tools.configuration import AICTConfig
    return AICTConfig.from_yaml('examples/config_energy.yaml')

@pytest.fixture(scope='session')
def cta_config():
    from aict_tools.configuration import AICTConfig
    return AICTConfig.from_yaml('examples/cta_config.yaml')


def test_read_telescope_data_feature_gen(h5py_file, fact_config):
    from aict_tools.io import read_telescope_data
    columns = fact_config.energy.columns_to_read_train
    
    feature_gen_config = fact_config.energy.feature_generation
    df = read_telescope_data(h5py_file, fact_config, columns=columns, feature_generation_config=feature_gen_config)
    assert set(df.columns) == set(fact_config.energy.features) | set([fact_config.energy.target_column])

    # new column with name 'area' should exist after feature generation
    assert 'area' in df.columns

def test_read_data(h5py_file):
    from aict_tools.io import read_data
    df = read_data(h5py_file, 'events')
    assert 'run_id' in df.columns
    assert 'width' in df.columns


def test_read_data_tables(tables_file):
    from aict_tools.io import read_data
    
    df = read_data(tables_file, 'telescope_events')
    assert 'telescope_id' in df.columns

    df = read_data(tables_file, 'array_events')
    assert 'array_event_id' in df.columns


def test_append_column_tables(tables_file):
    from aict_tools.io import read_data
    from aict_tools.io import append_column_to_hdf5

    df = read_data(tables_file, 'telescope_events')

    new_column_name = 'foobar'
    random_data = np.random.normal(size=len(df))
    append_column_to_hdf5(tables_file, random_data, 'telescope_events',  new_column_name)

    assert new_column_name not in df.columns

            
    df = read_data(tables_file, 'telescope_events')
    assert new_column_name in df.columns


def test_read_chunks_tables_feature_gen(tables_file, cta_config):
    from aict_tools.io import read_telescope_data_chunked
    chunk_size = 125

    columns = cta_config.energy.columns_to_read_train
    fg = cta_config.energy.feature_generation
    generator = read_telescope_data_chunked(tables_file, cta_config, chunk_size, columns=columns, feature_generation_config=fg)
    for df, _, _ in generator:
        assert not df.empty
        assert set(df.columns) == set(cta_config.energy.features + ['array_event_id', 'run_id']) | set([cta_config.energy.target_column])




def test_read_chunks_tables(tables_file, cta_config):
    from aict_tools.io import read_telescope_data_chunked
    from aict_tools.io import read_data
    chunk_size = 125
    generator = read_telescope_data_chunked(tables_file, cta_config, chunk_size, ['width', 'length'])

    stops = []
    for df, _, stop in generator:
        stops.append(stop)
        assert not df.empty
        assert set(df.columns) == set(['width', 'length', 'array_event_id', 'run_id'])

    df = read_data(tables_file, 'telescope_events')
    assert stops[-1] == len(df)
    # first chunk shuld have the given chunksize
    assert stops[1] - stops[0] == chunk_size


def test_read_chunks(h5py_file, fact_config):
    from aict_tools.io import read_telescope_data_chunked
    from aict_tools.io import read_data

    chunk_size = 25
    generator = read_telescope_data_chunked(h5py_file, fact_config, chunk_size, ['width', 'length'])
    
    stops = []
    for df, _, stop in generator:
        assert not df.empty
        assert set(df.columns) == set(['width', 'length'])
        stops.append(stop)

    df = read_data(h5py_file, 'events')
    assert stops[-1] == len(df)
    # first chunk shuld have the given chunksize
    assert stops[1] - stops[0] == chunk_size


def test_append_column_tables_chunked():
    from aict_tools.io import read_data
    from aict_tools.io import append_column_to_hdf5

    with tempfile.TemporaryDirectory(prefix='aict_tools_test_') as d:

        filename = 'cta_tables_file.hdf5'
        shutil.copy(f'examples/{filename}', d)
        temp_file = os.path.join(d, filename)
        
        df = read_data(temp_file, 'telescope_events')

        new_column_name = 'foobar'
        random_data = np.random.normal(size=len(df))
        append_column_to_hdf5(temp_file, random_data, 'telescope_events',  new_column_name)

        assert new_column_name not in df.columns

        df = read_data(temp_file, 'telescope_events')
        assert new_column_name in df.columns


def test_append_column_h5py(h5py_file):
    from aict_tools.io import read_data
    from aict_tools.io import append_column_to_hdf5
 
    df = read_data(h5py_file, 'events')

    new_column_name = 'foobar'
    random_data = np.random.normal(size=len(df))
    append_column_to_hdf5(h5py_file, random_data, 'events', new_column_name)

    assert new_column_name not in df.columns

    df = read_data(h5py_file, 'events')
    assert new_column_name in df.columns