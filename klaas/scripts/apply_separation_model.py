import numpy as np
import click
import joblib
import yaml
import logging

from ..io import check_extension, read_data, write_data
from ..preprocessing import convert_to_float32


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('data_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('model_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_path', type=click.Path(exists=False, dir_okay=False))
@click.option('-k', '--key', help='HDF5 key for pandas or h5py hdf5')
def main(configuration_path, data_path, model_path, output_path, key):
    '''
    Apply loaded model to data.
    The cuts applied during model training will also be applied here.

    CONFIGURATION_PATH: Path to the config yaml file.

    DATA_PATH: path to the FACT data.

    MODEL_PATH: Path to the pickled model.

    OUTPUT_PATH: Path to the data with added prediction columns.
    '''
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger()

    check_extension(output_path)

    with open(configuration_path) as f:
        config = yaml.load(f)

    training_variables = config['training_variables']
    query = config.get('query')

    log.info('Loading model')
    model = joblib.load(model_path)
    log.info('Done')

    log.info('Loading data')
    df_data = read_data(data_path, key=key)
    log.info('Done')

    if query is not None:
        df_data = df_data.query()

    df_data[training_variables] = convert_to_float32(df_data[training_variables])

    valid = np.logical_not(df_data[training_variables].isnull().any(axis=1))
    if len(df_data.loc[valid]) < len(df_data):
        invalid_columns = df_data[training_variables].isnull().any(axis=0)
        log.warning(
            'Data contains not-predictable events.\n'
            'There are nan-values in columns: {}'.format(
                df_data[training_variables].columns.loc[invalid_columns]
            )
        )

    log.info('Predicting on data...')
    prediction = model.predict_proba(df_data.loc[valid, training_variables])

    df_data.loc[valid, 'signal_prediction'] = prediction[:, 1]
    df_data['signal_theta'] = df_data['Theta']
    df_data['signal_distance'] = df_data['Distance']

    if 'Theta' in training_variables:
        log.info('Theta used in model, redoing classification for off regions')
        thetas = df_data['Theta'].copy()
        distances = df_data['Distance'].copy()
        alphas = df_data['Alpha'].copy()

        for region in range(1, 6):
            log.info('Predicting off position {}'.format(region))
            theta_key = 'Theta_Off_{}'.format(region)
            distance_key = 'Distance_Off_{}'.format(region)
            alpha_key = 'Alpha_Off_{}'.format(region)
            df_data['Theta'] = df_data[theta_key]
            df_data['Distance'] = df_data[distance_key]
            df_data['Alpha'] = df_data[alpha_key]

            prediction = model.predict_proba(df_data.loc[valid, training_variables])
            df_data.loc[valid, 'background_prediction_{}'.format(region)] = prediction[:, 1]

        df_data['Distance'] = distances
        df_data['Theta'] = thetas
        df_data['Alphas'] = alphas

    log.info('Writing output')
    write_data(df_data, output_path)


if __name__ == '__main__':
    main()
