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


def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


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

    columns_to_delete = [
        'reconstructed_source_position',
        'theta',
        'theta_deg',
        'theta_rec_pos',
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
    model = joblib.load(model_path)
    log.info('Done')

    if n_jobs:
        model.n_jobs = n_jobs

    columns_to_read = training_variables.copy()
    generation_config = config.get('feature_generation')
    if generation_config:
        columns_to_read.extend(generation_config['needed_keys'])

    columns_to_read.append('source_position')
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

        source_pos = np.full((len(df_data), 2), np.nan)

        prediction = model.predict(df_data.loc[valid, training_variables])
        source_pos[valid, 0] = prediction[:, 0]
        source_pos[valid, 1] = prediction[:, 1]

        theta = euclidean_distance(
            source_pos[:, 0],
            source_pos[:, 1],
            df_data['source_position_0'].values,
            df_data['source_position_1'].values,
        )

        theta_offs = {}
        for i in range(1, 6):
            theta_offs[i] = euclidean_distance(
                source_pos[:, 0],
                source_pos[:, 1],
                df_data['anti_source_position_{}_0'.format(i)].values,
                df_data['anti_source_position_{}_1'.format(i)].values,
            )

        with h5py.File(data_path, 'r+') as f:
            if 'theta' in f[key].keys():

                n_existing = f[key]['theta'].shape[0]
                n_new = theta.shape[0]

                f[key]['theta'].resize(n_existing + n_new, axis=0)
                f[key]['theta'][start:end] = theta
                f[key]['theta_deg'].resize(n_existing + n_new, axis=0)
                f[key]['theta_deg'][start:end] = camera_distance_mm_to_deg(theta)
                f[key]['reconstructed_source_position'].resize(n_existing + n_new, axis=0)
                f[key]['reconstructed_source_position'][start:end, :] = source_pos

                for i in range(1, 6):
                    f[key]['theta_off_' + str(i)].resize(n_existing + n_new, axis=0)
                    f[key]['theta_off_' + str(i)][start:end] = theta_offs[i]

                    f[key]['theta_deg_off_' + str(i)].resize(n_existing + n_new, axis=0)
                    f[key]['theta_deg_off_' + str(i)][start:end] = camera_distance_mm_to_deg(theta_offs[i])

            else:
                f[key].create_dataset('theta', data=theta, maxshape=(None, ))
                f[key].create_dataset(
                    'theta_deg',
                    data=camera_distance_mm_to_deg(theta),
                    maxshape=(None, )
                )
                f[key].create_dataset(
                    'reconstructed_source_position',
                    data=source_pos,
                    maxshape=(None, 2),
                )
                for i in range(1, 6):
                    f[key].create_dataset(
                        'theta_off_' + str(i),
                        data=theta_offs[i],
                        maxshape=(None, ),
                    )
                    f[key].create_dataset(
                        'theta_deg_off_' + str(i),
                        data=camera_distance_mm_to_deg(theta_offs[i]),
                        maxshape=(None, ),
                    )


if __name__ == '__main__':
    main()
