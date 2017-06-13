import pandas as pd
import click
from sklearn import model_selection
from sklearn import metrics
from tqdm import tqdm
import numpy as np
import yaml
from sklearn import ensemble

from fact.io import write_data, read_data
from ..io import pickle_model
from ..preprocessing import convert_to_float32
from ..feature_generation import feature_generation

import logging


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('signal_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('predictions_path', type=click.Path(exists=False, dir_okay=False))
@click.argument('model_path', type=click.Path(exists=False, dir_okay=False))
@click.option('-k', '--key', help='HDF5 key for pandas or h5py hdf5', default='events')
@click.option('-v', '--verbose', help='Verbose log output', is_flag=True)
def main(configuration_path, signal_path, predictions_path, model_path, key, verbose):
    '''
    Train a regressor to reconstruct the source position on simulated gammas.
    Both pmml and pickle format are supported for the output.

    CONFIGURATION_PATH: Path to the config yaml file

    SIGNAL_PATH: Path to the signal data

    PREDICTIONS_PATH : path to the file where the mc predictions are stored.

    MODEL_PATH: Path to save the model to.
        Allowed extensions are .pkl and .pmml.
        If extension is .pmml, then both pmml and pkl file will be saved
    '''

    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    log = logging.getLogger()

    with open(configuration_path) as f:
        config = yaml.load(f)

    n_signal = config.get('n_signal')

    n_cross_validations = config['n_cross_validations']
    training_variables = config['training_variables']
    target_names = config.get(
        'target_names',
        ['source_position_0', 'source_position_1']
    )
    target_column = config.get('target_column', 'source_position')

    assert len(target_names) == 2

    regressor = eval(config['regressor'])

    columns_to_read = training_variables + [target_column]

    # Also read columns needed for feature generation
    generation_config = config.get('feature_generation')
    if generation_config:
        columns_to_read.extend(generation_config.get('needed_keys', []))

    log.info('Loading data')
    df = read_data(
        file_path=signal_path,
        key=key,
        columns=columns_to_read,
    )

    log.info('Total number of events: {}'.format(len(df)))

    # generate features if given in config
    if generation_config:
        gen_config = config['feature_generation']
        training_variables.extend(sorted(gen_config['features']))
        feature_generation(df, gen_config, inplace=True)

    df_train = convert_to_float32(df[training_variables])
    df_train.dropna(how='any', inplace=True)

    if n_signal:
        log.info('Sampling {} random events'.format(n_signal))
        df_train = df_train.sample(n_signal)

    log.info('Events after nan-dropping: {} '.format(len(df_train)))

    target = df[target_names].loc[df_train.index]

    log.info('Starting {} fold cross validation... '.format(n_cross_validations))
    scores = []
    cv_predictions = []

    kfold = model_selection.KFold(n_splits=n_cross_validations, shuffle=True)

    for fold, (train, test) in tqdm(enumerate(kfold.split(df_train.values))):

        cv_x_train, cv_x_test = df_train.values[train], df_train.values[test]
        cv_y_train, cv_y_test = target.values[train], target.values[test]

        regressor.fit(cv_x_train, cv_y_train)
        cv_y_prediction = regressor.predict(cv_x_test)

        scores.append(metrics.r2_score(cv_y_test, cv_y_prediction))

        cv_predictions.append(pd.DataFrame({
            'source_x': cv_y_test[:, 0],
            'source_y': cv_y_test[:, 1],
            'source_x_prediction': cv_y_prediction[:, 0],
            'source_y_prediction': cv_y_prediction[:, 1],
            'cv_fold': fold
        }))

    predictions_df = pd.concat(cv_predictions, ignore_index=True)

    log.info('writing predictions from cross validation')
    write_data(predictions_df, predictions_path)

    scores = np.array(scores)
    log.info('Cross validated R^2 scores: {}'.format(scores))
    log.info('Mean R^2 score from CV: {:0.4f} ± {:0.4f}'.format(
        scores.mean(), scores.std()
    ))

    log.info('Building new model on complete data set...')
    regressor.fit(df_train.values, target.values)

    log.info('Pickling model to {} ...'.format(model_path))
    pickle_model(
            regressor,
            feature_names=list(df_train.columns),
            model_path=model_path,
            label_text='estimated_energy',
    )


if __name__ == '__main__':
    main()