import pandas as pd
import click
from sklearn import model_selection
from sklearn import metrics
from tqdm import tqdm
import numpy as np

from fact.io import write_data
from ..io import save_model, read_telescope_data
from ..preprocessing import (
    convert_to_float32, calc_true_disp, convert_units, horizontal_to_camera,
)
from ..feature_generation import feature_generation
from ..configuration import AICTConfig
from ..logging import setup_logging


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('signal_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('predictions_path', type=click.Path(exists=False, dir_okay=False))
@click.argument('dxdy_model_path', type=click.Path(exists=False, dir_okay=False))
@click.option('-k', '--key', help='HDF5 key for h5py hdf5', default='events')
@click.option('-v', '--verbose', help='Verbose log output', is_flag=True)
def main(configuration_path, signal_path, predictions_path, dxdy_model_path, key, verbose):
    '''
    Train one learner to be able to reconstruct the source position.
    One regressor for multiple outputs (dx,dy).

    Both pmml and pickle format are supported for the output.

    CONFIGURATION_PATH: Path to the config yaml file

    SIGNAL_PATH: Path to the signal data

    PREDICTIONS_PATH : path to the file where the mc predictions are stored.

    DXDY_MODEL_PATH: Path to save the dxdy model to.

        Allowed extensions are .pkl and .pmml.
        If extension is .pmml, then both pmml and pkl file will be saved
    '''
    log = setup_logging(verbose=verbose)

    config = AICTConfig.from_yaml(configuration_path)
    model_config = config.dxdy

    np.random.seed(config.seed)

    dxdy_regressor = model_config.dxdy_regressor

    dxdy_regressor.random_state = config.seed

    log.info('Loading data')
    df = read_telescope_data(
        signal_path, config,
        model_config.columns_to_read_train,
        feature_generation_config=model_config.feature_generation,
        n_sample=model_config.n_signal
    )
    log.info('Total number of events: {}'.format(len(df)))

    log.info(
        'Using coordinate transformations for %s',
        model_config.coordinate_transformation
    )

    df = convert_units(df, model_config)
    source_x, source_y = horizontal_to_camera(df, model_config)
    
    log.info('Using projected disp: {}'.format(model_config.project_disp))

    df['true_dx'] = source_x - df[model_config.cog_x_column]
    df['true_dy'] = source_y - df[model_config.cog_y_column]

    # generate features if given in config
    if model_config.feature_generation:
        feature_generation(df, model_config.feature_generation, inplace=True)

    df_train = convert_to_float32(df[model_config.features])
    df_train.dropna(how='any', inplace=True)

    log.info('Events after nan-dropping: {} '.format(len(df_train)))

    target_dx = df['true_dx'].loc[df_train.index]
    target_dy = df['true_dy'].loc[df_train.index]

    # load optional columns if available to be able to make performance plots
    # vs true energy / size
    if config.true_energy_column is not None:
        true_energy = df.loc[df_train.index, config.true_energy_column].to_numpy()
    if config.size_column is not None:
        size = df.loc[df_train.index, config.size_column].to_numpy()


    log.info('Starting {} fold cross validation... '.format(
        model_config.n_cross_validations
    ))
    scores_dxdy = []
    cv_predictions = []

    kfold = model_selection.KFold(
        n_splits=model_config.n_cross_validations,
        shuffle=True,
        random_state=config.seed,
    )

    total = model_config.n_cross_validations
    for fold, (train, test) in enumerate(tqdm(kfold.split(df_train.values), total=total)):

        cv_x_train, cv_x_test = df_train.values[train], df_train.values[test]

        cv_dxdy_train = np.stack((target_dx.values[train], target_dy.values[train]), axis=1)
        cv_dxdy_test = np.stack((target_dx.values[test], target_dy.values[test]), axis=1)

        dxdy_regressor.fit(cv_x_train, cv_dxdy_train)
        cv_dxdy_prediction = dxdy_regressor.predict(cv_x_test)

        if model_config.log_target is True:
            cv_dxdy_test = np.exp(cv_dxdy_test)
            cv_dxdy_prediction = np.exp(cv_dxdy_prediction)

        scores_dxdy.append(metrics.r2_score(cv_dxdy_test, cv_dxdy_prediction))
        cv_df = pd.DataFrame({
            'dx': cv_dxdy_test[:,0],
            'dy': cv_dxdy_test[:,1],
            'dx_prediction': cv_dxdy_prediction[:,0],
            'dy_prediction': cv_dxdy_prediction[:,1],
            'cv_fold': fold,
        })
        if config.true_energy_column is not None:
            cv_df[config.true_energy_column] = true_energy[test]
        if config.size_column is not None:
            cv_df[config.size_column] = size[test]
        cv_predictions.append(cv_df)

    predictions_df = pd.concat(cv_predictions, ignore_index=True)

    log.info('writing predictions from cross validation')
    write_data(predictions_df, predictions_path, mode='w')

    scores_dxdy = np.array(scores_dxdy)
    log.info('Cross validated R^2 scores for dxdy: {}'.format(scores_dxdy))
    log.info('Mean R^2 score from CV: {:0.4f} ± {:0.4f}'.format(
        scores_dxdy.mean(), scores_dxdy.std()
    ))

    log.info('Building new model on complete data set...')
    # set random seed again to make sure different settings
    # for n_cross_validations don't change the final model
    np.random.seed(config.seed)
    dxdy_regressor.random_state = config.seed

    dxdy_regressor.fit(df_train.values, np.stack((target_dx.values, target_dy.values), axis=1))

    log.info('Pickling dxdy model to {} ...'.format(dxdy_model_path))
    save_model(
        dxdy_regressor,
        feature_names=list(df_train.columns),
        model_path=dxdy_model_path,
        label_text='dxdy',
    )


if __name__ == '__main__':
    main()
