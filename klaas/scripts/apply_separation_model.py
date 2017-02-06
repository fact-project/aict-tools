import numpy as np
import click
import joblib
import yaml
import logging

from ..io import check_extension, read_data, write_data


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('data_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('model_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_path', type=click.Path(exists=False, dir_okay=False))
def main(configuration_path, data_path, model_path, output_path):
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
    query = config['query']

    log.info('Loading model')
    model = joblib.load(model_path)

    log.info('Loading data')
    # sklearn needs float32 values. Overflows create -infs and infs.
    df_data = read_data(data_path, query=query)
    df_data[training_variables] = df_data[training_variables].astype('float32')

    df_data.replace(np.inf, np.finfo('float32').max, inplace=True)
    df_data.replace(-np.inf, np.finfo('float32').min, inplace=True)
    df_data.dropna(how='any', inplace=True)

    log.info('After dropping nans there are {} events left.'.format(len(df_data)))

    log.info('Predicting on data...')
    prediction = model.predict_proba(df_data[training_variables])
    df_data['signal_prediction'] = prediction[:, 1]
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
            prediction = model.predict_proba(df_data[training_variables])
            df_data['background_prediction_{}'.format(region)] = prediction[:, 1]

        df_data['Distance'] = distances
        df_data['Theta'] = thetas
        df_data['Alphas'] = alphas

    log.info('Writing output')
    write_data(df_data, output_path)


if __name__ == '__main__':
    main()
