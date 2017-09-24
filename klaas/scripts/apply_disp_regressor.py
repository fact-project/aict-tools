import click
import numpy as np
from sklearn.externals import joblib
import yaml
import logging
import h5py
from tqdm import tqdm

from fact.io import read_h5py_chunked
from fact.instrument import camera_distance_mm_to_deg
from ..preprocessing import convert_to_float32, check_valid_rows
from ..feature_generation import feature_generation
from ..io import append_to_h5py


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('data_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('disp_model_path', type=click.Path(exists=False, dir_okay=False))
@click.argument('sign_model_path', type=click.Path(exists=False, dir_okay=False))
@click.option('-k', '--key', help='HDF5 key for pandas or h5py hdf5', default='events')
@click.option('-n', '--n-jobs', type=int, help='Number of cores to use')
@click.option('-y', '--yes', help='Do not prompt for overwrites', is_flag=True)
@click.option('-v', '--verbose', help='Verbose log output', is_flag=True)
@click.option(
    '-N', '--chunksize', type=int,
    help='If given, only process the given number of events at once',
)
def main(configuration_path, data_path, disp_model_path, sign_model_path, key, chunksize, n_jobs, yes, verbose):
    '''
    Apply given model to data. Two columns are added to the file, energy_prediction
    and energy_prediction_std

    CONFIGURATION_PATH: Path to the config yaml file
    DATA_PATH: path to the FACT data in a h5py hdf5 file, e.g. erna_gather_fits output
    DISP_MODEL_PATH: Path to the pickled disp model.
    SIGN_MODEL_PATH: Path to the pickled sign model.
    '''
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    log = logging.getLogger()

    with open(configuration_path) as f:
        config = yaml.load(f)

    training_variables = config['training_variables']

    columns_to_delete = [
        'reconstructed_source_position',
        'theta',
        'theta_deg',
        'theta_rec_pos',
        'disp_prediction',
    ]
    for i in range(1, 6):
        columns_to_delete.extend([
            'theta_off_' + str(i),
            'theta_deg_off_' + str(i),
            'theta_off_rec_pos_' + str(i),
        ])

    with h5py.File(data_path, 'r+') as f:
        for column in columns_to_delete:
            if column in f[key].keys():
                if not yes:
                    click.confirm(
                        'Dataset "{}" exists in file, overwrite?'.format(column),
                        abort=True,
                    )
                    yes = True
                del f[key][column]

    log.info('Loading model')
    disp_model = joblib.load(disp_model_path)
    sign_model = joblib.load(sign_model_path)
    log.info('Done')

    if n_jobs:
        disp_model.n_jobs = n_jobs
        sign_model.n_jobs = n_jobs

    columns_to_read = training_variables.copy()
    generation_config = config.get('feature_generation')
    if generation_config:
        columns_to_read.extend(generation_config['needed_keys'])

    columns_to_read.extend(['source_position', 'cog_x', 'cog_y', 'delta'])
    columns_to_read.extend('anti_source_position_{}'.format(i) for i in range(1, 6))

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

        disp = disp_model.predict(df_data.loc[valid, training_variables])
        sign = sign_model.predict(df_data.loc[valid, training_variables])

        rec_pos = np.full((len(df_data), 2), np.nan)
        rec_pos[valid, 0] = df_data.cog_x + disp * np.cos(df_data.delta) * sign
        rec_pos[valid, 1] = df_data.cog_y + disp * np.sin(df_data.delta) * sign

        source_pos = df_data.loc[:, ['source_position_0', 'source_position_1']].values
        theta = np.linalg.norm(rec_pos - source_pos, axis=1)

        theta_offs = {}
        for i in range(1, 6):
            cols = ['anti_source_position_{}_{}'.format(i, j) for j in range(2)]
            off_pos = df_data.loc[:, cols].values
            theta_offs[i] = np.linalg.norm(rec_pos - off_pos, axis=1)

        with h5py.File(data_path, 'r+') as f:
            append_to_h5py(f, theta, key, 'theta')
            append_to_h5py(f, camera_distance_mm_to_deg(theta), key, 'theta_deg')
            append_to_h5py(f, rec_pos, key, 'reconstructed_source_position')
            append_to_h5py(f, disp * sign, key, 'disp_prediction')

            for i in range(1, 6):
                append_to_h5py(f, theta_offs[i], key, 'theta_off_' + str(i))
                col = 'theta_deg_off_' + str(i)
                append_to_h5py(f, camera_distance_mm_to_deg(theta_offs[i]),  key, col)


if __name__ == '__main__':
    main()
