import os
from sklearn.externals import joblib
from sklearn2pmml import sklearn2pmml, PMMLPipeline
import logging
import numpy as np
from .feature_generation import feature_generation
from fact.io import read_h5py, write_data
import pandas as pd
import h5py
import click

__all__ = ['pickle_model']


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


def pickle_model(classifier, feature_names, model_path, label_text='label'):
    p, extension = os.path.splitext(model_path)
    classifier.feature_names = feature_names

    if (extension == '.pmml'):
        joblib.dump(classifier, p + '.pkl', compress=4)

        pipeline = PMMLPipeline([
            ('classifier', classifier)
        ])
        pipeline.target_field = label_text
        pipeline.active_fields = np.array(feature_names)
        sklearn2pmml(pipeline, model_path)

    else:
        joblib.dump(classifier, model_path, compress=4)


class HDFColumnAppender():
    '''
    This is a context manager which can append columns to an existing hdf5 table
    in a chunkwise manner.

    The contex manager was introduced to handle hypothetical
    memory problems with pytables.

    For now we decided to drop pytables support.

    Parameters
    ----------
    path: str
        path to the hdf5 file
    table_name: str
        name of the table columns should be appended to
    '''
    def __init__(self, path, table_name):
        self.path = path
        self.table_name = table_name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def add_data(self, data, new_column_name, start, stop):
        '''
        Appends a column containing new data to existing table.

        Parameters
        ----------
        data: array-like
            the data to append
        new_column_name: str
            name of the new column to append
        start: int or None
            first row to replace in the file
        stop: int or None
            last event to replace in the file
        '''
        append_column_to_hdf5(self.path, data, self.table_name, new_column_name)




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
        table_name = f.require_group(table_name)  # create if not exists

        max_shape = list(array.shape)
        max_shape[0] = None
        if new_column_name not in table_name.keys():
            table_name.create_dataset(
                new_column_name,
                data=array,
                maxshape=tuple(max_shape),
            )
        else:
            n_existing = table_name[new_column_name].shape[0]
            n_new = array.shape[0]

            table_name[new_column_name].resize(n_existing + n_new, axis=0)
            table_name[new_column_name][n_existing:n_existing + n_new] = array