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
import tables
__all__ = ['pickle_model']


log = logging.getLogger(__name__)

# write_data(selected_array_events, path, key='array_events', use_h5py=use_h5py, mode='a')
def write_hdf(data, path, table_name, mode='w', use_h5py='h5py', **kwargs):
    if use_h5py:
        write_data(data, path, key=table_name, use_h5py=True, mode=mode, **kwargs)
    else:
        with pd.HDFStore(path, mode) as storer:
            storer.put(table_name, data, format='t', append=(mode in ['a', 'r+']), **kwargs)


def get_number_of_rows_in_table(path, key):
    try:
        with h5py.File(path, 'r') as f:
            group = f.get(key)
            nrows = group[next(iter(group.keys()))].shape[0]
    
    except AttributeError:
        with pd.HDFStore(path, 'r') as storer:
            nrows = storer.get_storer(key).nrows

    return nrows


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
    try:
        with pd.HDFStore(path, 'r') as storer:
            names = storer.select(table_name, stop=0).columns.values
    except TypeError:
        with h5py.File(path, 'r') as f:
            names = list(f[table_name].keys())
    return names


def remove_column_from_file(path, table_name, column_to_remove):
    '''
    Removes a column from a hdf5 file. In case of 'tables' format needs to copy the entire table.
    '''
    try:
        with pd.HDFStore(path, 'r+') as store:
            df = store.select(table_name)
            df.drop(columns=[column_to_remove], inplace=True) 
            store.remove(table_name)
            store.put(table_name, df, format='t')
    except TypeError:
        with h5py.File(path, 'r+') as f:
            del f[table_name][column_to_remove]

def is_sorted(values, stable=False):
    i = 1 if stable else 0
    return (np.diff(values) >= i).all()

def has_holes(values):
    return (np.diff(values) > 1).any()

def read_telescope_data(path, aict_config, columns=None, feature_generation_config=None, n_sample=None, first=None, last=None):
    '''
    Read given columns from data and perform a random sample if n_sample is supplied.
    Returns a single pandas data frame
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
    This is a ContextManager which can append columns to an existing hdf5 table 
    in a chunkwise manner.
    For hdf5 files in *tables format* this will temprarily occupy twice the disk space.
    
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
        try:
            with pd.HDFStore(path, mode='r') as r:
                _ = r[table_name]
            self.is_tables_format = True
        except TypeError:
            self.is_tables_format = False

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.is_tables_format:
            with tables.open_file(self.path, 'r+') as t:
                try:
                    t.remove_node(f'/{self.table_name}', recursive='force')
                    t.rename_node(f'/{self.table_name}_copy', newname=self.table_name)
                except tables.exceptions.NoSuchNodeError:
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
        if self.is_tables_format:
            with pd.HDFStore(self.path, 'r+') as store:
                df = store.select(self.table_name, start=start, stop=stop)
                df[new_column_name] = data
                # store.remove(self.table_name, start=0, stop=stop-start)
                store.put(self.table_name + '_copy', df, format='t', append=True)
        else:
            _append_column_to_h5py(self.path, data, self.table_name, new_column_name)



def append_column_to_hdf5(path, array, table_name, new_column_name):
    '''
    Add array as a column to the hdf5 file. This needs to load the 
    entire table into memory if the hdf5 file is in 'tables' format.
    '''
    try:
        with pd.HDFStore(path, 'r+') as store:
            df = store.select(table_name)
            df[new_column_name] = array
            store.remove(table_name)
            store.put(table_name, df, format='t')
            
    except TypeError:
        _append_column_to_h5py(path, array, table_name, new_column_name)


def _append_column_to_h5py(path, array, table_name, new_column_name):
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