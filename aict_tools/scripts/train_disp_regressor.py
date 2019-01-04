import pandas as pd
import click
from sklearn import model_selection
from sklearn import metrics
from tqdm import tqdm
import numpy as np

from fact.io import write_data
from fact.coordinates.utils import horizontal_to_camera
from ..io import pickle_model, read_telescope_data
from ..preprocessing import convert_to_float32, calc_true_disp
from ..feature_generation import feature_generation
from ..configuration import AICTConfig

import logging


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

    config = AICTConfig.from_yaml(configuration_path)
    model_config = config.disp

    np.random.seed(config.seed)

    disp_regressor = model_config.disp_regressor
    sign_classifier = model_config.sign_classifier

    disp_regressor.random_state = config.seed
    sign_classifier.random_state = config.seed

    log.info('Loading data')
    df = read_telescope_data(
        signal_path, config,
        model_config.columns_to_read_train,
        feature_generation_config=model_config.feature_generation,
        n_sample=model_config.n_signal
    )
    log.info('Total number of events: {}'.format(len(df)))

    source_x, source_y = horizontal_to_camera(
        az=df[model_config.source_az_column],
        zd=df[model_config.source_zd_column],
        az_pointing=df[model_config.pointing_az_column],
        zd_pointing=df[model_config.pointing_zd_column],
    )

    df['true_disp'], df['true_sign'] = calc_true_disp(
        source_x, source_y,
        df[model_config.cog_x_column], df[model_config.cog_y_column],
        df[model_config.delta_column],
    )

    # generate features if given in config
    if model_config.feature_generation:
        feature_generation(df, model_config.feature_generation, inplace=True)

    df_train = convert_to_float32(df[config.disp.features])
    df_train.dropna(how='any', inplace=True)

    log.info('Events after nan-dropping: {} '.format(len(df_train)))

    target_disp = df['true_disp'].loc[df_train.index]
    target_sign = df['true_sign'].loc[df_train.index]

    log.info('Starting {} fold cross validation... '.format(
        model_config.n_cross_validations
    ))
    scores_disp = []
    scores_sign = []
    cv_predictions = []

    kfold = model_selection.KFold(
        n_splits=model_config.n_cross_validations,
        shuffle=True,
        random_state=config.seed,
    )

    total = model_config.n_cross_validations
    for fold, (train, test) in enumerate(tqdm(kfold.split(df_train.values), total=total)):

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
    # set random seed again to make sure different settings
    # for n_cross_validations don't change the final model
    np.random.seed(config.seed)
    disp_regressor.random_state = config.seed
    sign_classifier.random_state = config.seed

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
