import numpy as np
import click
from sklearn.externals import joblib
import yaml
import logging
import h5py
from tqdm import tqdm

from fact.io import read_h5py_chunked

from ..preprocessing import convert_to_float32, check_valid_rows
from ..feature_generation import feature_generation
from ..io import append_to_h5py


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('data_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('model_path', type=click.Path(exists=True, dir_okay=False))
@click.option('-k', '--key', help='HDF5 key for pandas or h5py hdf5', default='events')
@click.option('-n', '--n-jobs', type=int, help='Number of cores to use')
@click.option('-y', '--yes', help='Do not prompt for overwrites', is_flag=True)
@click.option('-v', '--verbose', help='Verbose log output', is_flag=True)
@click.option(
    '-N', '--chunksize', type=int,
    help='If given, only process the given number of events at once',
)
def main(configuration_path, data_path, model_path, key, chunksize, n_jobs, yes, verbose):
    '''
    Apply given model to data. Two columns are added to the file, energy_prediction
    and energy_prediction_std

    CONFIGURATION_PATH: Path to the config yaml file
    DATA_PATH: path to the FACT data in a h5py hdf5 file, e.g. erna_gather_fits output
    MODEL_PATH: Path to the pickled model
    '''
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    log = logging.getLogger()

    with open(configuration_path) as f:
        config = yaml.load(f)

    training_variables = config['training_variables']
    log_target = config.get('log_target', False)
    class_name = config.get('class_name', 'gamma_energy') + '_prediction'

    with h5py.File(data_path, 'r+') as f:
        if class_name in f[key].keys():
            if not yes:
                click.confirm(
                    'Dataset "{}" exists in file, overwrite?'.format(class_name),
                    abort=True,
                )
            del f[key][class_name]
            del f[key]['{}_std'.format(class_name)]

    log.info('Loading model')
    model = joblib.load(model_path)
    log.info('Done')

    if n_jobs:
        model.n_jobs = n_jobs

    columns_to_read = training_variables.copy()
    generation_config = config.get('feature_generation')
    if generation_config:
        columns_to_read.extend(generation_config['needed_keys'])

    df_generator = read_h5py_chunked(
        data_path,
        key=key,
        columns=columns_to_read,
        chunksize=chunksize,
    )

    if generation_config:
        training_variables.extend(sorted(generation_config['features']))

    log.info('Predicting on data...')
    for df_data, start, end in tqdm(df_generator):

        if generation_config:
            feature_generation(
                df_data,
                generation_config,
                inplace=True,
            )

        df_data[training_variables] = convert_to_float32(df_data[training_variables])
        valid = check_valid_rows(df_data[training_variables])

        energy_prediction = np.full(len(df_data), np.nan)
        energy_prediction_std = np.full(len(df_data), np.nan)
        predictions = np.array([
            t.predict(df_data.loc[valid, training_variables])
            for t in model.estimators_
        ])

        if log_target is True:
            predictions = np.exp(predictions)

        # this is equivalent to  model.predict(df_data[training_variables])
        energy_prediction[valid.values] = np.mean(predictions, axis=0)
        # also store the standard deviation in the table
        energy_prediction_std[valid.values] = np.std(predictions, axis=0)

        with h5py.File(data_path, 'r+') as f:
            append_to_h5py(f, energy_prediction, key, class_name)
            append_to_h5py(f, energy_prediction_std, key, class_name + '_std')


if __name__ == '__main__':
    main()
