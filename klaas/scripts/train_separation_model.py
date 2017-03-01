import pandas as pd
import click
from sklearn import model_selection
from tqdm import tqdm
import numpy as np
from sklearn import metrics
import yaml
import logging
from sklearn import ensemble

from ..io import read_data, pickle_model, write_data, check_extension
from ..preprocessing import convert_to_float32


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('signal_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('background_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('predictions_path', type=click.Path(exists=False, dir_okay=False))
@click.argument('model_path', type=click.Path(exists=False, dir_okay=False))
@click.option('-k', '--key', help='HDF5 key for pandas or h5py hdf5')
def main(configuration_path, signal_path, background_path, predictions_path, model_path, key):
    '''
    Train a classifier on signal and background monte carlo data and write the model
    to MODEL_PATH in pmml or pickle format.

    CONFIGURATION_PATH: Path to the config yaml file

    BACKGROUND_PATH: Path to the background data

    SIGNAL_PATH: Path to the signal data

    PREDICTIONS_PATH : path to the file where the mc predictions are stored.

    MODEL_PATH: Path to save the model to. Allowed extensions are .pkl and .pmml.
        If extension is .pmml, then both pmml and pkl file will be saved
    '''

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger()

    with open(configuration_path) as f:
        config = yaml.load(f)

    n_background = config.get('n_background')
    n_signal = config.get('n_signal')

    n_cross_validations = config.get('n_cross_validations', 10)

    training_variables = config['training_variables']

    classifier = eval(config['classifier'])

    check_extension(predictions_path)
    check_extension(model_path, allowed_extensions=['.pmml', '.pkl'])

    log.info('Loading signal data')
    df_signal = read_data(file_path=signal_path, key=key)
    df_signal['label_text'] = 'signal'
    df_signal['label'] = 1

    if n_signal is not None:
        log.info('Randomly sample {} events'.format(n_signal))
        df_signal = df_signal.sample(n_signal)

    log.info('Loading background data')
    df_background = read_data(file_path=background_path, key=key)
    df_background['label_text'] = 'background'
    df_background['label'] = 0

    if n_background is not None:
        log.info('Randomly sample {} events'.format(n_background))
        df_background = df_background.sample(n_background)


    df_full = pd.concat([df_background, df_signal], ignore_index=True)

    df_training = convert_to_float32(df_full[training_variables])
    log.info('Total training events: {}'.format(len(df_training)))

    df_training.dropna(how='any', inplace=True)
    log.info('Training events after dropping nans: {}'.format(len(df_training)))

    label = df_full.loc[df_training.index, 'label']

    n_gammas = len(label[label == 1])
    n_protons = len(label[label == 0])
    log.info('Training classifier with {} protons and {} gammas'.format(
        n_protons, n_gammas
    ))

    # save prediction_path for each cv iteration
    cv_predictions = []
    # iterate over test and training sets
    X = df_training.values
    y = label.values
    log.info('Starting {} fold cross validation... '.format(n_cross_validations))

    stratified_kfold = model_selection.StratifiedKFold(
        n_splits=n_cross_validations, shuffle=True,
    )

    aucs = []
    for fold, (train, test) in enumerate(tqdm(stratified_kfold.split(X, y), total=n_cross_validations)):
        # select data
        xtrain, xtest = X[train], X[test]
        ytrain, ytest = y[train], y[test]
        # fit and predict
        classifier.fit(xtrain, ytrain)

        y_probas = classifier.predict_proba(xtest)[:, 1]
        y_prediction = classifier.predict(xtest)
        cv_predictions.append(pd.DataFrame({
            'label': ytest,
            'label_prediction': y_prediction,
            'probabilities': y_probas,
            'cv_fold': fold
        }))
        aucs.append(metrics.roc_auc_score(ytest, y_probas))

    log.info('Mean AUC ROC : {}'.format(np.array(aucs).mean()))

    predictions_df = pd.concat(cv_predictions, ignore_index=True)
    log.info('writing predictions from cross validation')
    write_data(predictions_df, predictions_path)

    log.info('Training model on complete dataset')
    classifier.fit(X, y)

    log.info('Pickling model to {} ...'.format(model_path))
    pickle_model(
        classifier=classifier,
        model_path=model_path,
        label_text='label',
        feature_names=list(df_training.columns)
    )


if __name__ == '__main__':
    main()
