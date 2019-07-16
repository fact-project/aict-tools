import os
from sklearn.externals import joblib
import logging
import numpy as np
import pandas as pd
import h5py
import click

from fact.io import read_h5py, write_data

from .feature_generation import feature_generation
from . import __version__


__all__ = [
    'drop_prediction_column',
    'read_telescope_data',
    'read_telescope_data_chunked',
    'save_model',
]


log = logging.getLogger(__name__)


def write_hdf(data, path, table_name, mode='w', **kwargs):
    write_data(data, path, key=table_name, use_h5py=True, mode=mode, **kwargs)


def get_number_of_rows_in_table(path, key):

    with h5py.File(path, 'r') as f:
        group = f.get(key)
        return group[next(iter(group.keys()))].shape[0]


def read_data(file_path, key=None, columns=None, first=None, last=None, **kwargs):
    '''
    This is similar to the read_data function in fact.io
    pandas hdf5:   pd.HDFStore
    h5py hdf5:     fact.io.read_h5py
    '''
    _, extension = os.path.splitext(file_path)

    if extension in ['.hdf', '.hdf5', '.h5']:
        try:
            df = pd.read_hdf(file_path, key=key, columns=columns, start=first, stop=last, **kwargs)
        except (TypeError, ValueError):
            df = read_h5py(file_path, key=key, columns=columns, first=first, last=last, **kwargs)
        return df
    else:
        raise NotImplementedError(f'AICT tools cannot handle data with extension {extension} yet.')


def drop_prediction_column(data_path, group_name, column_name, yes=True):
    '''
    Deletes prediction columns in a h5py file if the columns exist.
    Including 'mean' and 'std' columns.
    '''
    with h5py.File(data_path, 'r+') as f:

        if group_name not in f.keys():
            return

        columns = f[group_name].keys()
        if column_name in columns:
            if not yes:
                click.confirm(
                    f'Column \"{column_name}\" exists in file, overwrite?', abort=True,
                )

            del f[group_name][column_name]

        if column_name + '_std' in columns:
            del f[group_name][column_name + '_std']
        if column_name + '_mean' in columns:
            del f[group_name][column_name + '_mean']


def read_telescope_data_chunked(path, aict_config, chunksize, columns=None, feature_generation_config=None):
    '''
    Reads data from hdf5 file given as PATH and yields merged datafrmes with feature generation applied for each chunk
    '''
    return TelescopeDataIterator(
        path,
        aict_config,
        chunksize,
        columns,
        feature_generation_config=feature_generation_config,
    )


def read_data_chunked(path, table_name, chunksize, columns=None):
    '''
    Reads data from hdf5 file given as PATH and yields dataframes for each chunk
    '''
    return HDFDataIterator(
        path,
        table_name,
        chunksize,
        columns,
    )


class HDFDataIterator:
    def __init__(
        self,
        path,
        table_name,
        chunksize,
        columns,
    ):
        self.path = path
        self.table_name = table_name
        self.n_rows = get_number_of_rows_in_table(path, table_name)
        self.columns = columns
        if chunksize:
            self.chunksize = chunksize
            self.n_chunks = int(np.ceil(self.n_rows / chunksize))
        else:
            self.n_chunks = 1
            self.chunksize = self.n_rows
        log.info('Splitting data into {} chunks'.format(self.n_chunks))

        self._current_chunk = 0

    def __len__(self):
        return self.n_chunks

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_chunk == self.n_chunks:
            raise StopIteration

        chunk = self._current_chunk
        start = chunk * self.chunksize
        end = min(self.n_rows, (chunk + 1) * self.chunksize)
        self._current_chunk += 1
        df = read_data(
            self.path,
            key=self.table_name,
            columns=self.columns,
            first=start,
            last=end,
        )

        return df, start, end


class TelescopeDataIterator:

    def __init__(
        self,
        path,
        aict_config,
        chunksize,
        columns,
        feature_generation_config=None,
    ):
        self.aict_config = aict_config
        self.columns = columns
        self.feature_generation_config = feature_generation_config
        if aict_config.has_multiple_telescopes:
            self.n_rows = get_number_of_rows_in_table(path, aict_config.array_events_key)
        else:
            self.n_rows = get_number_of_rows_in_table(path, aict_config.telescope_events_key)
        self.path = path
        if chunksize:
            self.chunksize = chunksize
            self.n_chunks = int(np.ceil(self.n_rows / chunksize))
        else:
            self.n_chunks = 1
            self.chunksize = self.n_rows
        log.info('Splitting data into {} chunks'.format(self.n_chunks))

        self._current_chunk = 0
        self._index_start = 0

    def __len__(self):
        return self.n_chunks

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_chunk == self.n_chunks:
            raise StopIteration

        chunk = self._current_chunk
        start = chunk * self.chunksize
        end = min(self.n_rows, (chunk + 1) * self.chunksize)
        self._current_chunk += 1
        df = read_telescope_data(
            self.path,
            aict_config=self.aict_config,
            columns=self.columns,
            first=start,
            last=end
        )

        df.index = np.arange(self._index_start, self._index_start + len(df))

        index_start = self._index_start
        self._index_start += len(df)
        index_stop = self._index_start

        if self.feature_generation_config:
            feature_generation(df, self.feature_generation_config, inplace=True)

        return df, index_start, index_stop


def get_column_names_in_file(path, table_name):
    '''Returns the list of column names in the given group

    Parameters
    ----------
    path : str
        path to hdf5 file
    table_name : str
        name of group/table in file

    Returns
    -------
    list
        list of column names
    '''
    with h5py.File(path, 'r') as f:
        return list(f[table_name].keys())


def remove_column_from_file(path, table_name, column_to_remove):
    '''
    Removes a column from a hdf5 file.

    Note: this is one of the reasons why we decided to not support pytables.
    In case of 'tables' format this needs to copy the entire table into memory and then some.


    Parameters
    ----------
    path : str
        path to hdf5 file
    table_name : str
        name of the group/table from which the column should be removed
    column_to_remove : str
        name of column to remove
    '''
    with h5py.File(path, 'r+') as f:
        del f[table_name][column_to_remove]


def is_sorted(values, stable=False):
    i = 1 if stable else 0
    return (np.diff(values) >= i).all()


def has_holes(values):
    return (np.diff(values) > 1).any()


def read_telescope_data(path, aict_config, columns=None, feature_generation_config=None, n_sample=None, first=None, last=None):
    '''    Read columns from data in file given under PATH.
        Returns a single pandas data frame containing all the requested data

    Parameters
    ----------
    path : str
        path to the hdf5 file to read
    aict_config : AICTConfig
        The configuration object. This is needed for gathering the primary keys to merge merge on.
    columns : list, optional
        column names to read, by default None
    feature_generation_config : FeatureGenerationConfig, optional
        The configuration object containing the information for feature generation, by default None
    n_sample : int, optional
        number of rows to randomly sample from the file, by default None
    first : int, optional
        first row to read from file, by default None
    last : int, optional
        last row to read form file, by default None

    Returns
    -------
    pd.DataFrame
        Dataframe containing the requested data.

    '''
    telescope_event_columns = None
    array_event_columns = None
    if aict_config.has_multiple_telescopes:
        join_keys = [aict_config.run_id_column, aict_config.array_event_id_column]

        if columns:
            t = aict_config.array_events_key
            array_event_columns = get_column_names_in_file(path, table_name=t)
            t = aict_config.telescope_events_key
            telescope_event_columns = get_column_names_in_file(path, table_name=t)

            array_event_columns = set(array_event_columns) & set(columns)
            telescope_event_columns = set(telescope_event_columns) & set(columns)
            array_event_columns |= set(join_keys)
            telescope_event_columns |= set(join_keys)

        tel_event_index = read_data(
            file_path=path,
            key=aict_config.telescope_events_key,
            columns=['run_id', 'array_event_id', 'width'],
        ).reset_index(drop=True)
        array_event_index = read_data(
            file_path=path,
            key=aict_config.array_events_key,
            columns=['run_id', 'array_event_id', 'num_triggered_telescopes'],
        ).reset_index(drop=True).iloc[first:last]

        tel_event_index['index_in_file'] = tel_event_index.index
        r = pd.merge(array_event_index, tel_event_index, left_on=join_keys, right_on=join_keys)

        # these asserts have been added to catch weird effects on old pandas version (< 0.20).
        # I'll leave them here in case this changes again with new version. as the consequences were quite subtle
        assert is_sorted(r.index_in_file)
        assert not has_holes(r.index_in_file)
        telescope_events = read_data(
            file_path=path,
            key=aict_config.telescope_events_key,
            columns=telescope_event_columns,
            first=r.index_in_file.iloc[0],
            last=r.index_in_file.iloc[-1] + 1,
        )
        array_events = read_data(
            file_path=path,
            key=aict_config.array_events_key,
            columns=array_event_columns,
        )
        df = pd.merge(left=array_events, right=telescope_events, left_on=join_keys, right_on=join_keys)
        assert len(df) == len(telescope_events)
        assert len(df) == len(r)

    else:
        df = read_data(
            file_path=path,
            key=aict_config.telescope_events_key,
            columns=columns,
            first=first,
            last=last,
        )

    if n_sample is not None:
        if n_sample > len(df):
            raise ValueError(
                'number of sampled events'
                ' {} must be smaller than number events in file {} ({})'
                .format(n_sample, path, len(df))
            )
        log.info('Randomly sample {} events'.format(n_sample))
        state = np.random.RandomState()
        state.set_state(np.random.get_state())
        df = df.sample(n_sample, random_state=state)

    # generate features if given in config
    if feature_generation_config:
        feature_generation(df, feature_generation_config, inplace=True)

    return df


def save_model(model, feature_names, model_path, label_text='label'):
    p, extension = os.path.splitext(model_path)
    model.feature_names = feature_names
    pickle_path = p + '.pkl'

    if extension == '.pmml':
        try:
            from sklearn2pmml import sklearn2pmml, PMMLPipeline
        except ImportError:
            raise ImportError(
                'You need to install `sklearn2pmml` to store models in pmml format'
            )

        pipeline = PMMLPipeline([
            ('model', model)
        ])
        pipeline.target_field = label_text
        pipeline.active_fields = np.array(feature_names)
        sklearn2pmml(pipeline, model_path)

    elif extension == '.onnx':

        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
            from onnx.onnx_pb import StringStringEntryProto
        except ImportError:
            raise ImportError(
                'You need to install `skl2onnx` to store models in onnx format'
            )

        onnx = convert_sklearn(
            model,
            name=label_text,
            initial_types=[('input', FloatTensorType((1, len(feature_names))))],
            doc_string='Model created by aict-tools to estimate {}'.format(label_text),
        )
        metadata = dict(
            model_author='aict-tools',
            aict_tools_version=__version__,
            feature_names=','.join(feature_names),
        )
        for key, value in metadata.items():
            onnx.metadata_props.append(StringStringEntryProto(key=key, value=value))

        # this makes sure we only get the scores and that they are numpy arrays and not
        # a list of dicts
        if hasattr(model, 'predict_proba'):
            onnx = select_model_inputs_outputs(onnx, ['probabilities'])

        with open(model_path, 'wb') as f:
            f.write(onnx.SerializeToString())
    else:
        pickle_path = model_path

    # Always store the pickle dump,just in case
    joblib.dump(model, pickle_path, compress=4)


def append_column_to_hdf5(path, array, table_name, new_column_name):
    '''
    Add array of values as a new column to the given file.

    Parameters
    ----------
    path : str
        path to file
    array : array-like
        values to append to the file
    table_name : str
        name of the group to append to
    new_column_name : str
        name of the new column
    '''
    with h5py.File(path, 'r+') as f:
        group = f.require_group(table_name)  # create if not exists

        max_shape = list(array.shape)
        max_shape[0] = None
        if new_column_name not in group.keys():
            group.create_dataset(
                new_column_name,
                data=array,
                maxshape=tuple(max_shape),
            )
        else:
            n_existing = group[new_column_name].shape[0]
            n_new = array.shape[0]

            group[new_column_name].resize(n_existing + n_new, axis=0)
            group[new_column_name][n_existing:n_existing + n_new] = array


def set_sample_fraction(path, fraction):
    with h5py.File(path, mode='r+') as f:
        before = f.attrs.get('sample_fraction', 1.0)
        f.attrs['sample_fraction'] = before * fraction


def copy_runs_group(inpath, outpath):
    with h5py.File(inpath, mode='r+') as infile, h5py.File(outpath) as outfile:
        for key in ('runs', 'corsika_runs'):
            if key in infile:
                log.info('Copying group "{}"'.format(key))
                infile.copy(key, outfile)
