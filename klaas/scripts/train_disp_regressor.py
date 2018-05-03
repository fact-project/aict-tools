import pandas as pd
import click
from sklearn import model_selection
from sklearn import metrics
from tqdm import tqdm
import numpy as np
import yaml
from sklearn import ensemble

from fact.io import write_data, read_data
from fact.coordinates.utils import horizontal_to_camera
from ..io import pickle_model
from ..preprocessing import convert_to_float32
from ..feature_generation import feature_generation
from ..features import find_used_source_features

import logging


def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('signal_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('predictions_path', type=click.Path(exists=False, dir_okay=False))
@click.argument('disp_model_path', type=click.Path(exists=False, dir_okay=False))
@click.argument('sign_model_path', type=click.Path(exists=False, dir_okay=False))
@click.option('-k', '--key', help='HDF5 key for h5py hdf5', default='events')
@click.option('-v', '--verbose', help='Verbose log output', is_flag=True)
def main(configuration_path, signal_path, predictions_path, disp_model_path, sign_model_path, key, verbose):
    '''
    Train two learners to be able to reconstruct the source position.
    One regressor for disp and one classifier for the sign of delta.

    Both pmml and pickle format are supported for the output.

    CONFIGURATION_PATH: Path to the config yaml file

    SIGNAL_PATH: Path to the signal data

    PREDICTIONS_PATH : path to the file where the mc predictions are stored.

    DISP_MODEL_PATH: Path to save the disp model to.

    SIGN_MODEL_PATH: Path to save the disp model to.
        Allowed extensions are .pkl and .pmml.
        If extension is .pmml, then both pmml and pkl file will be saved
    '''

    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    log = logging.getLogger()

    with open(configuration_path) as f:
        config = yaml.load(f)
    model_config = config.get('disp', config)

    seed = config.get('seed', 0)

    np.random.seed(seed)

    n_signal = model_config.get('n_signal')

    n_cross_validations = model_config.get('n_cross_validations', config.get('n_cross_validations', 5))
    training_variables = model_config['training_variables']

    disp_regressor = eval(model_config['disp_regressor'])
    sign_classifier = eval(model_config['sign_classifier'])

    disp_regressor.random_state = seed
    sign_classifier.random_state = seed

    az_source_col = model_config.get('source_azimuth_column', 'az_source')
    zd_source_col = model_config.get('source_zenith_column', 'zd_source')
    az_pointing_col = model_config.get('pointing_azimuth_column', 'az_tracking')
    zd_pointing_col = model_config.get('pointing_zenith_column', 'zd_tracking')

    columns_to_read = training_variables + [
        'cog_x', 'cog_y', 'delta',
        az_source_col, zd_source_col,
        az_pointing_col, zd_pointing_col
    ]

    # Also read columns needed for feature generation
    generation_config = model_config.get('feature_generation')
    if generation_config:
        columns_to_read.extend(generation_config.get('needed_keys', []))

    if len(find_used_source_features(training_variables, generation_config)) > 0:
        raise click.ClickException(
            'Using source dependent features in the model is not supported'
        )

    log.info('Loading data')
    df = read_data(
        file_path=signal_path,
        key=key,
        columns=columns_to_read,
    )
    log.info('Total number of events: {}'.format(len(df)))

    source_x, source_y = horizontal_to_camera(
        az=df[az_source_col], zd=df[zd_source_col],
        az_pointing=df[az_pointing_col], zd_pointing=df[zd_pointing_col],
    )

    df['true_disp'] = euclidean_distance(
        source_x, source_y,
        df.cog_x, df.cog_y
    )

    true_delta = np.arctan2(
        df.cog_y - source_y,
        df.cog_x - source_x,
    )
    df['true_sign'] = np.sign(np.abs(df.delta - true_delta) - np.pi / 2)

    # generate features if given in config
    if generation_config:
        training_variables.extend(sorted(generation_config['features']))
        feature_generation(df, generation_config, inplace=True)

    df_train = convert_to_float32(df[training_variables])
    df_train.dropna(how='any', inplace=True)

    if n_signal:
        log.info('Sampling {} random events'.format(n_signal))
        df_train = df_train.sample(n_signal, random_state=seed)

    log.info('Events after nan-dropping: {} '.format(len(df_train)))

    target_disp = df['true_disp'].loc[df_train.index]
    target_sign = df['true_sign'].loc[df_train.index]

    log.info('Starting {} fold cross validation... '.format(n_cross_validations))
    scores_disp = []
    scores_sign = []
    cv_predictions = []

    kfold = model_selection.KFold(
        n_splits=n_cross_validations,
        shuffle=True,
        random_state=seed,
    )

    for fold, (train, test) in tqdm(enumerate(kfold.split(df_train.values))):

        cv_x_train, cv_x_test = df_train.values[train], df_train.values[test]

        cv_disp_train, cv_disp_test = target_disp.values[train], target_disp.values[test]
        cv_sign_train, cv_sign_test = target_sign.values[train], target_sign.values[test]

        disp_regressor.fit(cv_x_train, cv_disp_train)
        cv_disp_prediction = disp_regressor.predict(cv_x_test)

        sign_classifier.fit(cv_x_train, cv_sign_train)
        cv_sign_prediction = sign_classifier.predict(cv_x_test)

        scores_disp.append(metrics.r2_score(cv_disp_test, cv_disp_prediction))
        scores_sign.append(metrics.accuracy_score(cv_sign_test, cv_sign_prediction))

        cv_predictions.append(pd.DataFrame({
            'disp': cv_disp_test,
            'disp_prediction': cv_disp_prediction,
            'sign': cv_sign_test,
            'sign_prediction': cv_sign_prediction,
            'cv_fold': fold
        }))

    predictions_df = pd.concat(cv_predictions, ignore_index=True)

    log.info('writing predictions from cross validation')
    write_data(predictions_df, predictions_path, mode='w')

    scores_disp = np.array(scores_disp)
    scores_sign = np.array(scores_sign)
    log.info('Cross validated R^2 scores for disp: {}'.format(scores_disp))
    log.info('Mean R^2 score from CV: {:0.4f} ± {:0.4f}'.format(
        scores_disp.mean(), scores_disp.std()
    ))

    log.info('Cross validated accuracy for the sign: {}'.format(scores_sign))
    log.info('Mean accuracy from CV: {:0.4f} ± {:0.4f}'.format(
        scores_sign.mean(), scores_sign.std()
    ))

    log.info('Building new model on complete data set...')
    disp_regressor.fit(df_train.values, target_disp.values)
    sign_classifier.fit(df_train.values, target_sign.values)

    log.info('Pickling disp model to {} ...'.format(disp_model_path))
    pickle_model(
        disp_regressor,
        feature_names=list(df_train.columns),
        model_path=disp_model_path,
        label_text='disp',
    )
    log.info('Pickling sign model to {} ...'.format(sign_model_path))
    pickle_model(
        sign_classifier,
        feature_names=list(df_train.columns),
        model_path=sign_model_path,
        label_text='disp',
    )


if __name__ == '__main__':
    main()
