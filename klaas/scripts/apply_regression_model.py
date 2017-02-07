import numpy as np
import click
from sklearn.externals import joblib
import yaml
import logging

from ..io import check_extension, read_data, write_data
from ..preprocessing import convert_to_float32, check_valid_rows


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('data_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('model_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('predictions_path', type=click.Path(exists=False, dir_okay=False))
@click.option('-k', '--key', help='HDF5 key for pandas or h5py hdf5')
@click.option(
    '-N', '--chunksize', type=int,
    help='If given, only process the given number of events at once',
)
def main(configuration_path, data_path, model_path, predictions_path, key, chunksize):
    '''
    Apply given model to data. Two columns are added to the file, energy_prediction
    and energy_prediction_std

    CONFIGURATION_PATH: Path to the config yaml file
    DATA_PATH: path to the FACT data in a h5py hdf5 file, e.g. erna_gather_fits output
    MODEL_PATH: Path to the pickled model
    PREDICTIONS_PATH: Path to the data with added prediction columns
    '''
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger()

    check_extension(predictions_path)

    with open(configuration_path) as f:
        config = yaml.load(f)

    training_variables = config['training_variables']

    log.info('Loading model')
    model = joblib.load(model_path)
    log.info('Done')

    log.info('Loading data')

    if chunksize is not None:
    df_data = read_data(data_path, key=key)
    df_data[training_variables] = convert_to_float32(df_data[training_variables])

    valid = check_valid_rows(df_data[training_variables])

    log.info('After query there are {} events left.'.format(len(df_data)))
    log.info('Predicting on data...')
    predictions = np.array([
        t.predict(df_data.loc[valid, training_variables])
        for t in model.estimators_
    ])

    # this is equivalent to  model.predict(df_data[training_variables])
    df_data.loc[valid, 'energy_prediction'] = np.mean(predictions, axis=0)
    # also store the standard deviation in the table
    df_data.loc[valid, 'energy_prediction_std'] = np.std(predictions, axis=0)


if __name__ == '__main__':
    main()
