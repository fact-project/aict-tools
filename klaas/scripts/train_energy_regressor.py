import pandas as pd
import click
from sklearn import model_selection
from sklearn import metrics
from tqdm import tqdm
import numpy as np

from fact.io import write_data
from ..io import pickle_model, read_and_sample_data
from ..preprocessing import convert_to_float32
from ..feature_generation import feature_generation
from ..features import has_source_dependent_columns
from ..configuration import KlaasConfig
import logging

logging.basicConfig()
log = logging.getLogger()


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('signal_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('predictions_path', type=click.Path(exists=False, dir_okay=False))
@click.argument('model_path', type=click.Path(exists=False, dir_okay=False))
@click.option('-k', '--key', help='HDF5 key for h5py hdf5', default='events')
@click.option('-v', '--verbose', help='Verbose log output', is_flag=True)
def main(configuration_path, signal_path, predictions_path, model_path, key, verbose):
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
    logging.getLogger().setLevel(logging.DEBUG if verbose else logging.INFO)

    config = KlaasConfig(configuration_path)

    if has_source_dependent_columns(config.columns_to_read):
        raise click.ClickException(
            'Using source dependent features in the model is not supported'
        )

    df = read_and_sample_data(signal_path, config, n_sample=config.training_config.n_signal)

    log.info('Total number of events: {}'.format(len(df)))

    # generate features if given in config
    if config.feature_generation_config:
        feature_generation(df, config.feature_generation_config, inplace=True)

    df_train = convert_to_float32(df[config.training_variables])
    df_train.dropna(how='any', inplace=True)

    log.info('Events after nan-dropping: {} '.format(len(df_train)))

    target = df[config.target_name].loc[df_train.index]
    target.name = 'true_energy'

    if config.log_target is True:
        target = np.log(target)

    n_cross_validations = config.training_config.n_cross_validations
    regressor = config.training_config.model
    log.info('Starting {} fold cross validation... '.format(n_cross_validations))
    scores = []
    cv_predictions = []

    kfold = model_selection.KFold(n_splits=n_cross_validations, shuffle=True, random_state=config.seed)

    for fold, (train, test) in tqdm(enumerate(kfold.split(df_train.values))):

        cv_x_train, cv_x_test = df_train.values[train], df_train.values[test]
        cv_y_train, cv_y_test = target.values[train], target.values[test]

        regressor.fit(cv_x_train, cv_y_train)
        cv_y_prediction = regressor.predict(cv_x_test)

        if config.log_target is True:
            cv_y_test = np.exp(cv_y_test)
            cv_y_prediction = np.exp(cv_y_prediction)

        scores.append(metrics.r2_score(cv_y_test, cv_y_prediction))

        cv_predictions.append(pd.DataFrame({
            'label': cv_y_test,
            'label_prediction': cv_y_prediction,
            'cv_fold': fold
        }))

    predictions_df = pd.concat(cv_predictions, ignore_index=True)

    log.info('writing predictions from cross validation')
    write_data(predictions_df, predictions_path, mode='w')

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
