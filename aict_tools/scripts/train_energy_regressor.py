import pandas as pd
import click
from sklearn import model_selection
from sklearn import metrics
from tqdm import tqdm
import numpy as np

from fact.io import write_data
from ..io import pickle_model, read_telescope_data
from ..preprocessing import convert_to_float32
from ..configuration import AICTConfig
import logging

logging.basicConfig()
log = logging.getLogger()


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('signal_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('predictions_path', type=click.Path(exists=False, dir_okay=False))
@click.argument('model_path', type=click.Path(exists=False, dir_okay=False))
@click.option('-v', '--verbose', help='Verbose log output', is_flag=True)
def main(configuration_path, signal_path, predictions_path, model_path, verbose):
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

    config = AICTConfig.from_yaml(configuration_path)
    model_config = config.energy

    df = read_telescope_data(
        signal_path, config, model_config.columns_to_read_train,
        feature_generation_config=model_config.feature_generation,
        n_sample=model_config.n_signal
    )

    log.info('Total number of events: {}'.format(len(df)))

    df_train = convert_to_float32(df[model_config.features])
    df_train.dropna(how='any', inplace=True)

    log.debug('Events after nan-dropping: {} '.format(len(df_train)))

    target = df[model_config.target_column].loc[df_train.index]
    target.name = 'true_energy'

    if model_config.log_target is True:
        target = np.log(target)

    n_cv = model_config.n_cross_validations
    regressor = model_config.model
    log.info('Starting {} fold cross validation... '.format(n_cv))
    scores = []
    cv_predictions = []

    kfold = model_selection.KFold(n_splits=n_cv, shuffle=True, random_state=config.seed)

    for fold, (train, test) in enumerate(tqdm(kfold.split(df_train.values), total=n_cv)):

        cv_x_train, cv_x_test = df_train.values[train], df_train.values[test]
        cv_y_train, cv_y_test = target.values[train], target.values[test]

        regressor.fit(cv_x_train, cv_y_train)
        cv_y_prediction = regressor.predict(cv_x_test)

        if model_config.log_target is True:
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
    # set random seed again to make sure different settings
    # for n_cross_validations don't change the final model
    np.random.seed(config.seed)
    regressor.random_state = config.seed

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
