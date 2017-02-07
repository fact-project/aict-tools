import pandas as pd
import click
from sklearn import model_selection
from sklearn import metrics
from tqdm import tqdm
import numpy as np
import yaml
from sklearn import ensemble

from .. import write_data, pickle_model, read_data
from ..preprocessing import convert_to_float32

import logging


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('signal_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('predictions_path', type=click.Path(exists=False, dir_okay=False))
@click.argument('model_path', type=click.Path(exists=False, dir_okay=False))
@click.option('-k', '--key', help='HDF5 key for pandas or h5py hdf5')
def main(configuration_path, signal_path, predictions_path, model_path, key):
    '''
    Train an energy regressor simulated gamma.
    Both pmml and pickle format are supported for the output.

    CONFIGURATION_PATH: Path to the config yaml file

    SIGNAL_PATH: Path to the signal data

    PREDICTIONS_PATH : path to the file where the mc predictions are stored.

    MODEL_PATH: Path to save the model to.
        Allowed extensions are .pkl and .pmml.
        If extension is .pmml, then both pmml and pkl file will be saved
    '''

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger()

    with open(configuration_path) as f:
        config = yaml.load(f)

    query = config.get('query')
    n_signal = config.get('n_signal')

    n_cross_validations = config['n_cross_validations']
    training_variables = config['training_variables']

    classifier = eval(config['classifier'])

    log.info('Loading data')
    df = read_data(file_path=signal_path, key=key)

    log.info('Total number of events: {}'.format(len(df)))

    df_train = convert_to_float32(df[training_variables])
    df_train.dropna(how='any', inplace=True)

    if query:
        log.info('Using query: {}'.format(query))
        df_train = df_train.query(query)

    if n_signal:
        log.info('Sampling {} random events'.format(n_signal))
        df_train = df_train.sample(n_signal)

    log.info('Events after nan-dropping: {} '.format(len(df_train)))

    target = df['MCorsikaEvtHeader.fTotalEnergy'].loc[df_train.index]
    target.name = 'true_energy'

    log.info('Starting {} fold cross validation... '.format(n_cross_validations))
    scores = []
    cv_predictions = []

    kfold = model_selection.KFold(n_splits=n_cross_validations, shuffle=True)

    for fold, (train, test) in tqdm(enumerate(kfold.split(df_train.values))):

        cv_x_train, cv_x_test = df_train.values[train], df_train.values[test]
        cv_y_train, cv_y_test = target.values[train], target.values[test]

        classifier.fit(cv_x_train, cv_y_train)
        cv_y_prediction = classifier.predict(cv_x_test)

        scores.append(metrics.r2_score(cv_y_test, cv_y_prediction))

        cv_predictions.append(pd.DataFrame({
            'label': cv_y_test,
            'label_prediction': cv_y_prediction,
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
    classifier.fit(df_train.values, target.values)

    log.info('Pickling model to {} ...'.format(model_path))
    pickle_model(
            classifier=classifier,
            feature_names=list(df_train.columns),
            model_path=model_path,
            label_text='estimated_energy',
    )


if __name__ == '__main__':
    main(o)
