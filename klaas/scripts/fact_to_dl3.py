import click
import numpy as np
from sklearn.externals import joblib
import yaml
import logging
from tqdm import tqdm
import pandas as pd
from functools import partial

from astropy.time import Time
from astropy.coordinates import AltAz, SkyCoord
import astropy.units as u

from fact.io import read_h5py, read_h5py_chunked, to_h5py
from fact.instrument.constants import LOCATION
from fact.analysis.source import calc_theta_camera, calc_theta_offs_camera
from fact.coordinates import camera_to_equatorial

from ..apply import predict_energy, predict_disp, predict_separator
from ..parallel import parallelize_array_computation


def to_altaz(obstime, source):
    altaz = AltAz(location=LOCATION, obstime=obstime)
    return source.transform_to(altaz)


def concat_results_altaz(results):
    obstime = np.concatenate([s.obstime for s in results])
    return SkyCoord(
        alt=np.concatenate([s.alt.deg for s in results]) * u.deg,
        az=np.concatenate([s.az.deg for s in results]) * u.deg,
        frame=AltAz(location=LOCATION, obstime=obstime)
    )


def calc_source_features_sim(
    source_x,
    source_y,
    source_zd,
    source_az,
    pointing_position_zd,
    pointing_position_az,
):
    result = {}
    result['theta_deg'] = calc_theta_camera(
        source_x,
        source_y,
        source_zd=source_zd,
        source_az=source_az,
        zd_pointing=pointing_position_zd,
        az_pointing=pointing_position_az,
    )
    theta_offs = calc_theta_offs_camera(
        source_x,
        source_y,
        source_zd=source_zd,
        source_az=source_az,
        zd_pointing=pointing_position_zd,
        az_pointing=pointing_position_az,
        n_off=5,
    )
    for i, theta_off in enumerate(theta_offs, start=1):
        result['theta_deg_off_{}'.format(i)] = theta_off
    return result


def calc_source_features_obs(
    source_x,
    source_y,
    source_zd,
    source_az,
    pointing_position_zd,
    pointing_position_az,
    obstime,
):

    result = calc_source_features_sim(
        source_x,
        source_y,
        source_zd,
        source_az,
        pointing_position_zd,
        pointing_position_az,
    )

    result['ra_prediction'], result['dec_prediction'] = camera_to_equatorial(
        source_x,
        source_y,
        pointing_position_zd,
        pointing_position_az,
        obstime,
    )
    return result


dl3_columns = [
    'run_id',
    'event_num',
    'gamma_energy_prediction',
    'gamma_prediction',
    'theta_deg',
    'theta_deg_off_1',
    'theta_deg_off_2',
    'theta_deg_off_3',
    'theta_deg_off_4',
    'theta_deg_off_5',
]
corsika_columns = [
    'corsika_run_header_run_number',
    'corsika_event_header_event_number',
    'ceres_event_event_reuse',
    'corsika_event_header_num_reuse',
    'corsika_event_header_total_energy',
    'corsika_event_header_x',
    'corsika_event_header_y',
    'corsika_event_header_first_interaction_height',
]
dl3_columns_sim = dl3_columns + corsika_columns

dl3_columns_obs = dl3_columns + [
    'night',
    'ra_prediction',
    'dec_prediction',
    'timestamp'
]

needed_columns = [
    'cog_x', 'cog_y', 'delta', 'pointing_position_az', 'pointing_position_zd',
    'run_id', 'event_num',
]


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('data_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('separator_model_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('energy_model_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('disp_model_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('sign_model_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('output', type=click.Path(exists=False, dir_okay=False))
@click.option('-k', '--key', help='HDF5 key for h5py hdf5', default='events')
@click.option('-n', '--n-jobs', default=-1, type=int, help='Number of cores to use')
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
    output,
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

    columns = set(needed_columns)
    for model in ('separator', 'energy', 'disp'):
        columns.update(config[model]['training_variables'])

        generation_config = config[model].get('feature_generation')
        if generation_config:
            columns.update(generation_config['needed_keys'])

    try:
        runs = read_h5py(data_path, key='runs')
        sources = runs['source'].unique()
        if len(sources) > 1:
            raise click.ClickException(
                'to_dl3 only supports files with a single source'
            )
        source = SkyCoord.from_name(sources[0])
        columns.update(['timestamp', 'night'])
    except (KeyError, OSError) as e:
        source = None
        columns.update(['source_position_az', 'source_position_zd'])
        columns.update(corsika_columns)

    df_generator = read_h5py_chunked(
        data_path,
        key=key,
        columns=columns,
        chunksize=chunksize,
        mode='r+'
    )

    log.info('Predicting on data...')
    for df, start, end in tqdm(df_generator):
        df['gamma_prediction'] = predict_separator(
            df, separator_model, config['separator']
        )
        df['gamma_energy_prediction'] = predict_energy(
            df, energy_model, config['energy']
        )

        disp = predict_disp(
            df, disp_model, sign_model, config['disp']
        )

        source_x = df.cog_x + disp * np.cos(df.delta)
        source_y = df.cog_y + disp * np.sin(df.delta)
        df['source_x_prediction'] = source_x
        df['source_y_prediction'] = source_y

        if source:
            obstime = Time(pd.to_datetime(df['timestamp'].values).to_pydatetime())
            source_altaz = concat_results_altaz(parallelize_array_computation(
                partial(to_altaz, source=source),
                obstime,
                n_jobs=n_jobs,
            ))

            result = parallelize_array_computation(
                calc_source_features_obs,
                source_x,
                source_y,
                source_altaz.zen.deg,
                source_altaz.az.deg,
                df['pointing_position_zd'].values,
                df['pointing_position_az'].values,
                obstime,
                n_jobs=n_jobs,
            )
        else:
            result = parallelize_array_computation(
                calc_source_features_sim,
                source_x,
                source_y,
                df['source_position_zd'].values,
                df['source_position_az'].values,
                df['pointing_position_zd'].values,
                df['pointing_position_az'].values,
                n_jobs=n_jobs,
            )

        for k in result[0].keys():
            df[k] = np.concatenate([r[k] for r in result])

        if source:
            to_h5py(df[dl3_columns_obs], output, key='events', mode='a')
        else:
            to_h5py(df[dl3_columns_sim], output, key='events', mode='a')


if __name__ == '__main__':
    main()
