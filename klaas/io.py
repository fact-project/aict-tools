from os import path
from sklearn.externals import joblib
from sklearn2pmml import sklearn2pmml, PMMLPipeline
import logging
import numpy as np
from .feature_generation import feature_generation
from fact.io import read_data
import pandas as pd

__all__ = ['pickle_model']


log = logging.getLogger(__name__)


def read_and_sample_data(path, klaas_config, n_sample=None):
    '''
    Read given columns from data and perform a random sample if n_sample is supplied.
    Returns a single pandas data frame
    '''
    if klaas_config.has_multiple_telescopes:
        df_features = read_data(
            file_path=path,
            key=klaas_config.telescope_events_key,
        )
        df_array_events = read_data(
            file_path=path,
            key=klaas_config.array_events_key,
        )
        df = pd.merge(left=df_array_events, right=df_features, on=klaas_config.array_event_id_key)
        df = df[klaas_config.columns_to_read]
    else:
        df = read_data(
            file_path=path,
            key=klaas_config.telescope_events_key,
            columns=klaas_config.columns_to_read,
        )

    if n_sample is not None:
        if n_sample > len(df):
            log.error(
                'number of sampled events {} must be smaller than number events in file {} ({})'
                .format(n_sample, path, len(df))
            )
            raise ValueError
        log.info('Randomly sample {} events'.format(n_sample))
        df = df.sample(n_sample)

    # generate features if given in config
    if klaas_config.feature_generation_config:
        feature_generation(df, klaas_config.feature_generation_config, inplace=True)

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
