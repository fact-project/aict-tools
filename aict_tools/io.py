from os import path
from sklearn.externals import joblib
from sklearn2pmml import sklearn2pmml, PMMLPipeline
import logging
import numpy as np
from .feature_generation import feature_generation
from fact.io import read_data, h5py_get_n_rows
import pandas as pd
import h5py
import click
__all__ = ['pickle_model']


log = logging.getLogger(__name__)


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


def read_telescope_data_chunked(path, aict_config, chunksize, columns, feature_generation_config=None):
    '''
    Reads data from hdf5 file given as PATH and yields dataframes for each chunk
    '''
    n_rows = h5py_get_n_rows(path, aict_config.telescope_events_key)
    if chunksize:
        n_chunks = int(np.ceil(n_rows / chunksize))
    else:
        n_chunks = 1
        chunksize = n_rows
    log.info('Splitting data into {} chunks'.format(n_chunks))

    for chunk in range(n_chunks):

        start = chunk * chunksize
        end = min(n_rows, (chunk + 1) * chunksize)

        df = read_telescope_data(
            path,
            aict_config=aict_config,
            columns=columns,
            first=start,
            last=end
        )
        df.index = np.arange(start, end)

        if feature_generation_config:
            feature_generation(df, feature_generation_config, inplace=True)

        yield df, start, end


def read_telescope_data(path, aict_config, columns, feature_generation_config=None, n_sample=None, first=None, last=None):
    '''
    Read given columns from data and perform a random sample if n_sample is supplied.
    Returns a single pandas data frame
    '''
    telescope_event_columns = None
    array_event_columns = None
    if aict_config.has_multiple_telescopes:
        join_keys = [aict_config.run_id_column, aict_config.array_event_id_column]
        if columns:
            with h5py.File(path, 'r') as f:
                array_event_columns = set(f[aict_config.array_events_key].keys()) & set(columns)
                telescope_event_columns = set(f[aict_config.telescope_events_key].keys()) & set(columns)
                array_event_columns |= set(join_keys)
                telescope_event_columns |= set(join_keys)

        telescope_events = read_data(
            file_path=path,
            key=aict_config.telescope_events_key,
            columns=telescope_event_columns,
            first=first,
            last=last,
        )
        array_events = read_data(
            file_path=path,
            key=aict_config.array_events_key,
            columns=array_event_columns,
        )

        df = pd.merge(left=array_events, right=telescope_events, left_on=join_keys, right_on=join_keys)

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
    p, extension = path.splitext(model_path)
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


def append_to_h5py(f, array, group, key):
    '''
    Write numpy array to h5py hdf5 file
    '''
    group = f.require_group(group)  # create if not exists

    max_shape = list(array.shape)
    max_shape[0] = None

    if key not in group.keys():
        group.create_dataset(
            key,
            data=array,
            maxshape=tuple(max_shape),
        )
    else:
        n_existing = group[key].shape[0]
        n_new = array.shape[0]

        group[key].resize(n_existing + n_new, axis=0)
        group[key][n_existing:n_existing + n_new] = array
