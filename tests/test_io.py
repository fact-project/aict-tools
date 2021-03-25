import shutil
import numpy as np
import pytest
from aict_tools.configuration import AICTConfig


@pytest.fixture(scope='function')
def hdf5_file(tmpdir_factory, request):
    fn = tmpdir_factory.mktemp('aict_test_data').join('test_file.hdf5')
    shutil.copy('examples/gamma.hdf5', fn)
    return fn, 'events', AICTConfig.from_yaml('examples/config_energy.yaml')


@pytest.fixture(scope='function')
def cta_file(tmpdir_factory, request):
    fn = tmpdir_factory.mktemp('aict_test_data').join('cta_file_test.h5')
    shutil.copy('examples/cta_gammas_diffuse.dl1.h5', fn)
    return fn


@pytest.fixture(scope='session')
def fact_config():
    from aict_tools.configuration import AICTConfig
    return AICTConfig.from_yaml('examples/config_energy.yaml')


@pytest.fixture(scope='session')
def cta_config():
    from aict_tools.configuration import AICTConfig
    return AICTConfig.from_yaml('examples/cta_full_config.yaml')


def test_read_default_columns(hdf5_file):
    from aict_tools.io import read_data, get_column_names_in_file
    from pandas.testing import assert_frame_equal

    path, table_name, config = hdf5_file

    df = read_data(path, table_name)

    cols = get_column_names_in_file(path, table_name)
    df_all_columns = read_data(path, table_name, columns=cols)
    assert_frame_equal(df, df_all_columns)


def test_read_default_columns_chunked(hdf5_file):
    from aict_tools.io import read_telescope_data, read_telescope_data_chunked
    import pandas as pd
    from pandas.testing import assert_frame_equal

    path, table_name, config = hdf5_file

    generator = read_telescope_data_chunked(path, config, 100)
    df_chunked = pd.concat([df for df, _, _ in generator]).reset_index(drop=True)

    df = read_telescope_data(path, config).reset_index(drop=True)

    assert_frame_equal(df, df_chunked)


def test_read_chunks(hdf5_file):
    from aict_tools.io import read_telescope_data_chunked, read_telescope_data
    import pandas as pd
    from pandas.testing import assert_frame_equal

    path, table_name, config = hdf5_file
    cols = ['width', 'length', ]

    chunk_size = 125
    generator = read_telescope_data_chunked(path, config, chunk_size, cols)

    dfs = []
    for df, _, _ in generator:
        dfs.append(df)
        assert not df.empty

    df_chunked = pd.concat(dfs).reset_index(drop=True)
    df = read_telescope_data(path, config, columns=cols).reset_index(drop=True)
    assert_frame_equal(df, df_chunked)


def test_read_chunks_cta_dl1(cta_file, cta_config):
    from aict_tools.io import read_telescope_data, read_telescope_data_chunked
    import pandas as pd
    from pandas.testing import assert_frame_equal

    chunk_size = 500
    # choose some columns from different tables in the file
    columns = [
        'true_energy',
        'azimuth',
        'equivalent_focal_length',
        'hillas_width',
        'tel_id',
        'event_id',
        'obs_id'
    ]

    generator = read_telescope_data_chunked(
        cta_file,
        cta_config,
        chunk_size,
        columns=columns
    )
    df1 = pd.concat(
        [df for df, _, _ in generator]
    )
    df2 = read_telescope_data(
        cta_file,
        cta_config,
        columns=columns
    )
    assert_frame_equal(df1, df2)

    # make sure we only loaded the telescopes we wanted
    np.testing.assert_array_equal(
        df2.tel_id.unique(),
        [int(x.split('_')[1]) for x in cta_config.telescopes]
    )


def test_remove_column(hdf5_file):
    from aict_tools.io import get_column_names_in_file
    from aict_tools.io import remove_column_from_file

    path, table, _ = hdf5_file
    columns = get_column_names_in_file(path, table)
    assert 'width' in columns

    remove_column_from_file(path, table, 'width')
    columns = get_column_names_in_file(path, table)
    assert 'width' not in columns


def test_columns_in_file(hdf5_file):
    from aict_tools.io import get_column_names_in_file

    path, table_name, _ = hdf5_file
    columns = get_column_names_in_file(path, table_name)
    assert 'width' in columns
    assert 'length' in columns


def test_read_data(hdf5_file):
    from aict_tools.io import read_data

    path, _, _ = hdf5_file
    df = read_data(path, 'events')
    assert 'run_id' in df.columns
    assert 'width' in df.columns


def test_append_column(hdf5_file):
    from aict_tools.io import read_data
    from aict_tools.io import append_column_to_hdf5

    path, table_name, _ = hdf5_file
    new_column_name = 'foobar'

    df = read_data(path, table_name)
    assert new_column_name not in df.columns

    random_data = np.random.normal(size=len(df))
    append_column_to_hdf5(path, random_data, table_name, new_column_name)

    df = read_data(path, table_name)
    assert new_column_name in df.columns


def test_append_column_chunked(hdf5_file):
    from aict_tools.io import read_telescope_data_chunked, read_data
    from aict_tools.io import append_column_to_hdf5

    path, table_name, config = hdf5_file

    new_column_name = 'foobar'
    chunk_size = 125

    df = read_data(path, table_name)

    assert new_column_name not in df.columns

    columns = config.energy.columns_to_read_train

    generator = read_telescope_data_chunked(path, config, chunk_size, columns=columns)
    for df, start, stop in generator:
        assert not df.empty
        new_data = np.arange(start, stop, step=1)
        append_column_to_hdf5(path, new_data, table_name, new_column_name)

    df = read_data(path, table_name)

    assert new_column_name in df.columns
    assert np.array_equal(df.foobar, np.arange(0, len(df)))


def test_read_chunks_cta_feature_gen(cta_file, cta_config):
    from aict_tools.io import read_telescope_data_chunked

    chunk_size = 100

    columns = cta_config.energy.columns_to_read_train
    fg = cta_config.energy.feature_generation
    generator = read_telescope_data_chunked(
        cta_file, cta_config, chunk_size, columns=columns, feature_generation_config=fg
    )
    for df, _, _ in generator:
        assert not df.empty
        assert set(df.columns) == set(
            cta_config.energy.features
            + fg.needed_columns +
            ['obs_id', 'event_id', 'tel_id']
        ) | set([cta_config.energy.target_column])


def test_read_telescope_data_feature_gen(hdf5_file, fact_config):
    from aict_tools.io import read_telescope_data

    columns = fact_config.energy.columns_to_read_train
    path, _, _ = hdf5_file
    feature_gen_config = fact_config.energy.feature_generation
    df = read_telescope_data(
        path, fact_config, columns=columns, feature_generation_config=feature_gen_config
    )
    assert set(df.columns) == set(fact_config.energy.features) | set(
        [fact_config.energy.target_column]
    )

    # new column with name 'area' should exist after feature generation
    assert 'area' in df.columns
