from os import path
import pandas as pd
import json
from sklearn_pandas import DataFrameMapper
from sklearn.externals import joblib
from sklearn2pmml import sklearn2pmml
import h5py
import sys
import logging
import numpy as np

__all__ = [
    'write_data', 'to_native_byteorder', 'read_h5py', 'read_h5py_chunked',
    'read_pandas_hdf5', 'pickle_model', 'check_extension', 'read_data'
]

log = logging.getLogger(__name__)


allowed_extensions = ('.hdf', '.hdf5', '.h5', '.json', '.csv')
native_byteorder = native_byteorder = {'little': '<', 'big': '>'}[sys.byteorder]


def write_data(df, file_path, hdf_key='table'):

    name, extension = path.splitext(file_path)

    if extension in ['.hdf', '.hdf5', '.h5']:
        df.to_hdf(file_path, key=hdf_key)

    elif extension == '.json':
        df.to_json(file_path)

    elif extension == '.csv':
        df.to_csv(file_path, delimiter=',', index=False)

    else:
        raise IOError(
            'cannot write tabular data with format {}. Allowed formats: {}'.format(
                extension, 'hdf5, json, csv'
            )
        )


def to_native_byteorder(array):
    ''' Convert numpy array to native byteorder '''

    if array.dtype.byteorder not in ('=', native_byteorder):
        return array.byteswap().newbyteorder()

    return array


def read_h5py(file_path, key='events', columns=None):
    '''
    Read a hdf5 file written with h5py into a dataframe

    Parameters
    ----------
    file_path: str
        file to read in
    key: str
        name of the hdf5 group to read in
    columns: iterable[str]
        Names of the datasets to read in. If not given read all 1d datasets
    '''
    with h5py.File(file_path) as f:
        group = f.get(key)
        if group is None:
            raise IOError('File does not contain group "{}"'.format(key))

        # get all columns of which don't have more than one value per row
        if columns is None:
            columns = [col for col in group.keys() if group[col].ndim == 1]

        df = pd.DataFrame()
        for col in columns:
            df[col] = to_native_byteorder(group[col][:])

    return df


def h5py_get_n_events(file_path, key='events'):

    with h5py.File(file_path) as f:
        group = f.get(key)

        if group is None:
            raise IOError('File does not contain group "{}"'.format(key))

        return group[next(iter(group.keys()))].shape[0]


def read_h5py_chunked(file_path, key='events', columns=None, chunksize=None):
    '''
    Generator function to read from h5py hdf5 in chunks,
    returns an iterator over pandas dataframes.

    When chunksize is None, use 1 chunk
    '''
    with h5py.File(file_path) as f:
        group = f.get(key)
        if group is None:
            raise IOError('File does not contain group "{}"'.format(key))

        # get all columns of which don't have more than one value per row
        if columns is None:
            columns = [col for col in group.keys() if group[col].ndim == 1]

        n_events = h5py_get_n_events(file_path, key=key)

        if chunksize is None:
            n_chunks = 1
            chunksize = n_events
        else:
            n_chunks = int(np.ceil(n_events / chunksize))
            log.info('Splitting data into {} chunks'.format(n_chunks))

        for chunk in range(n_chunks):

            start = chunk * chunksize
            end = min(n_events, (chunk + 1) * chunksize)

            df = pd.DataFrame(index=np.arange(start, end))

            for col in columns:
                df[col] = to_native_byteorder(group[col][start:end])

            yield df, start, end


def read_pandas_hdf5(file_path, key=None, columns=None, chunksize=None):
    df = pd.read_hdf(file_path, key=key, columns=columns, chunksize=chunksize)
    return df


def read_data(file_path, key=None, columns=None):
    name, extension = path.splitext(file_path)

    if extension in ['.hdf', '.hdf5', '.h5']:
        try:
            df = read_pandas_hdf5(
                file_path,
                key=key or 'table',
                columns=columns,
            )
        except (TypeError, ValueError):

            df = read_h5py(
                file_path,
                key=key or 'events',
                columns=columns,
            )

    elif extension == '.json':
        with open(file_path, 'r') as j:
            d = json.load(j)
            df = pd.DataFrame(d)
    else:
        raise NotImplementedError('Unknown data file extension {}'.format(extension))

    return df


def check_extension(file_path, allowed_extensions=allowed_extensions):
    p, extension = path.splitext(file_path)
    if extension not in allowed_extensions:
        raise IOError('Allowed formats: {}'.format(allowed_extensions))


def pickle_model(classifier, feature_names, model_path, label_text='label'):
    p, extension = path.splitext(model_path)
    classifier.feature_names = feature_names
    if (extension == '.pmml'):
        print("Pickling model to {} ...".format(model_path))

        mapper = DataFrameMapper([
            (feature_names, None),
            (label_text, None),
        ])

        joblib.dump(classifier, p + '.pkl', compress=4)
        sklearn2pmml(classifier, mapper,  model_path)

    else:
        joblib.dump(classifier, model_path, compress=4)
