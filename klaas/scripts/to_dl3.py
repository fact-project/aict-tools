import click
import numpy as np
from sklearn.externals import joblib
import yaml
import logging
from tqdm import tqdm

from fact.io import read_h5py_chunked, to_h5py
from ..predict import predict_energy, predict_disp, predict_separator


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('data_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('separator_model_path', type=click.Path(exists=False, dir_okay=False))
@click.argument('energy_model_path', type=click.Path(exists=False, dir_okay=False))
@click.argument('disp_model_path', type=click.Path(exists=False, dir_okay=False))
@click.argument('sign_model_path', type=click.Path(exists=False, dir_okay=False))
@click.option('-k', '--key', help='HDF5 key for h5py hdf5', default='events')
@click.option('-n', '--n-jobs', type=int, help='Number of cores to use')
@click.option('-y', '--yes', help='Do not prompt for overwrites', is_flag=True)
@click.option('-v', '--verbose', help='Verbose log output', is_flag=True)
@click.option(
    '-N', '--chunksize', type=int,
    help='If given, only process the given number of events at once',
)
def main(
    configuration_path,
    data_path,
    separator_model_path,
    energy_model_path,
    disp_model_path,
    sign_model_path,
    key,
    chunksize,
    n_jobs,
    yes,
    verbose,
):
    '''
    Apply given model to data. Two columns are added to the file, energy_prediction
    and energy_prediction_std

    CONFIGURATION_PATH: Path to the config yaml file
    DATA_PATH: path to the FACT data in a h5py hdf5 file, e.g. erna_gather_fits output
    SEPARATOR_MODEL_PATH: Path to the pickled separation model.
    ENERGY_MODEL_PATH: Path to the pickled energy regression model.
    DISP_MODEL_PATH: Path to the pickled disp model.
    SIGN_MODEL_PATH: Path to the pickled sign model.
    '''
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    log = logging.getLogger()

    with open(configuration_path) as f:
        config = yaml.load(f)

    log.info('Loading model')
    separator_model = joblib.load(separator_model_path)
    energy_model = joblib.load(energy_model_path)
    disp_model = joblib.load(disp_model_path)
    sign_model = joblib.load(sign_model_path)
    log.info('Done')

    if n_jobs:
        separator_model.n_jobs = n_jobs
        energy_model.n_jobs = n_jobs
        disp_model.n_jobs = n_jobs
        sign_model.n_jobs = n_jobs

    columns = set()
    for key in ('separator', 'energy_regressor', 'disp_regressor'):
        columns.update(config[key]['training_variables'])

        generation_config = config[key].get('feature_generation')
        if generation_config:
            columns.update(generation_config['needed_keys'])

    columns.update(['cog_x', 'cog_y', 'delta'])

    df_generator = read_h5py_chunked(
        data_path,
        key=key,
        columns=columns,
        chunksize=chunksize,
        mode='r+'
    )

    log.info('Predicting on data...')
    for df_data, start, end in tqdm(df_generator):

        df_data['gamma_prediction'] = predict_separator(
            df_data, separator_model, config['separator']
        )
        df_data['energy_predictio'] = predict_energy(
            df_data, energy_model, config['energy_regressor']
        )

        disp = predict_disp(
            df_data, disp_model, sign_model, config['disp_regressor']
        )

        source_x = df_data.cog_x + disp * np.cos(df_data.delta)
        source_y = df_data.cog_y + disp * np.sin(df_data.delta)


if __name__ == '__main__':
    main()

