import click
import numpy as np
from sklearn.externals import joblib
import logging
from tqdm import tqdm
import pandas as pd
from functools import partial
import os
import h5py

from astropy.time import Time
from astropy.coordinates import AltAz, SkyCoord
import astropy.units as u

from fact.io import read_h5py, to_h5py
from fact.instrument.constants import LOCATION
from fact.analysis.source import calc_theta_camera, calc_theta_offs_camera
from fact.coordinates import (
    camera_to_equatorial, horizontal_to_camera, camera_to_horizontal,
)
from fact.instrument import camera_distance_mm_to_deg

from ..apply import predict_energy, predict_disp, predict_separator
from ..parallel import parallelize_array_computation
from ..io import read_telescope_data_chunked
from ..configuration import AICTConfig
from ..feature_generation import feature_generation
from ..preprocessing import calc_true_disp


dl3_columns = [
    'run_id',
    'event_num',
    'gamma_energy_prediction',
    'gamma_prediction',
    'disp_prediction',
    'theta_deg',
    'theta_deg_off_1',
    'theta_deg_off_2',
    'theta_deg_off_3',
    'theta_deg_off_4',
    'theta_deg_off_5',
    'pointing_position_az',
    'pointing_position_zd',
]
dl3_columns_sim_read = [
    'corsika_run_header_run_number',
    'corsika_event_header_event_number',
    'ceres_event_event_reuse',
    'corsika_event_header_num_reuse',
    'corsika_event_header_total_energy',
    'corsika_event_header_x',
    'corsika_event_header_y',
    'corsika_event_header_first_interaction_height',
    'source_position_az',
    'source_position_zd',
]
dl3_columns_sim = dl3_columns_sim_read + dl3_columns + ['true_disp']

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


def calc_random_source(pointing_zd, pointing_az, wobble_distance):
    phi = np.random.uniform(0, 2 * np.pi, len(pointing_zd))

    r = wobble_distance / camera_distance_mm_to_deg(1)
    x = r * np.cos(phi)
    y = r * np.sin(phi)

    zd, az = camera_to_horizontal(x, y, pointing_zd, pointing_az)

    return zd, az


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


def calc_source_features_common(
    prediction_x,
    prediction_y,
    source_zd,
    source_az,
    pointing_position_zd,
    pointing_position_az,
):
    result = {}
    result['theta_deg'] = calc_theta_camera(
        prediction_x,
        prediction_y,
        source_zd=source_zd,
        source_az=source_az,
        zd_pointing=pointing_position_zd,
        az_pointing=pointing_position_az,
    )

    theta_offs = calc_theta_offs_camera(
        prediction_x,
        prediction_y,
        source_zd=source_zd,
        source_az=source_az,
        zd_pointing=pointing_position_zd,
        az_pointing=pointing_position_az,
        n_off=5,
    )
    for i, theta_off in enumerate(theta_offs, start=1):
        result['theta_deg_off_{}'.format(i)] = theta_off
    return result


def calc_source_features_sim(
    prediction_x,
    prediction_y,
    source_zd,
    source_az,
    pointing_position_zd,
    pointing_position_az,
    cog_x,
    cog_y,
    delta,
):
    result = calc_source_features_common(
        prediction_x,
        prediction_y,
        source_zd,
        source_az,
        pointing_position_zd,
        pointing_position_az,
    )
    source_x, source_y = horizontal_to_camera(
        az=source_az,
        zd=source_zd,
        az_pointing=pointing_position_az,
        zd_pointing=pointing_position_zd,
    )

    true_disp, true_sign = calc_true_disp(
        source_x, source_y,
        cog_x, cog_y,
        delta,
    )
    result['true_disp'] = true_disp * true_sign

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

    result = calc_source_features_common(
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


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('data_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('separator_model_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('energy_model_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('disp_model_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('sign_model_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('output', type=click.Path(exists=False, dir_okay=False))
@click.option('--random-source', help='Draw a random source position')
@click.option('--wobble-distance', help='Wobble distance in degree for random source position', default=0.6)
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
    random_source,
    wobble_distance,
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

    config = AICTConfig.from_yaml(configuration_path)

    if os.path.isfile(output):
        if not yes:
            click.confirm(
                'Outputfile {} exists. Overwrite?'.format(output),
                abort=True,
            )
        open(output, 'w').close()

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
        model_config = getattr(config, model)
        columns.update(model_config.columns_to_read_apply)
    try:
        runs = read_h5py(data_path, key='runs')
        sources = runs['source'].unique()
        if len(sources) > 1:
            raise click.ClickException(
                'to_dl3 only supports files with a single source'
            )
        source = SkyCoord.from_name(sources[0])
        columns.update(['timestamp', 'night'])
    except (KeyError, OSError):
        source = None
        columns.update(dl3_columns_sim_read)

    df_generator = read_telescope_data_chunked(
        data_path,
        config,
        chunksize=chunksize,
        columns=columns,
    )

    log.info('Predicting on data...')
    for df, start, end in tqdm(df_generator):
        df_sep = feature_generation(df, config.separator.feature_generation)
        df['gamma_prediction'] = predict_separator(
            df_sep[config.separator.features], separator_model,
        )

        df_energy = feature_generation(df, config.energy.feature_generation)
        df['gamma_energy_prediction'] = predict_energy(
            df_energy[config.energy.features],
            energy_model,
            log_target=config.energy.log_target,
        )

        df_disp = feature_generation(df, config.disp.feature_generation)
        disp = predict_disp(
            df_disp[config.disp.features], disp_model, sign_model
        )

        prediction_x = df.cog_x + disp * np.cos(df.delta)
        prediction_y = df.cog_y + disp * np.sin(df.delta)
        df['source_x_prediction'] = prediction_x
        df['source_y_prediction'] = prediction_y
        df['disp_prediction'] = disp

        if source:
            obstime = Time(pd.to_datetime(df['timestamp'].values).to_pydatetime())
            source_altaz = concat_results_altaz(parallelize_array_computation(
                partial(to_altaz, source=source),
                obstime,
                n_jobs=n_jobs,
            ))

            result = parallelize_array_computation(
                calc_source_features_obs,
                prediction_x,
                prediction_y,
                source_altaz.zen.deg,
                source_altaz.az.deg,
                df['pointing_position_zd'].values,
                df['pointing_position_az'].values,
                obstime,
                n_jobs=n_jobs,
            )
        else:

            if random_source:
                zd, az = calc_random_source(
                    df['pointing_position_zd'],
                    df['pointing_position_az'],
                    wobble_distance,
                )
                df['source_position_zd'] = zd
                df['source_position_az'] = az

            result = parallelize_array_computation(
                calc_source_features_sim,
                prediction_x,
                prediction_y,
                df['source_position_zd'].values,
                df['source_position_az'].values,
                df['pointing_position_zd'].values,
                df['pointing_position_az'].values,
                df['cog_x'].values,
                df['cog_y'].values,
                df['delta'].values,
                n_jobs=n_jobs,
            )

        for k in result[0].keys():
            df[k] = np.concatenate([r[k] for r in result])

        if source:
            to_h5py(df[dl3_columns_obs], output, key='events', mode='a')
        else:
            to_h5py(df[dl3_columns_sim], output, key='events', mode='a')

    with h5py.File(data_path, 'r') as f:
        sample_fraction = f.attrs.get('sample_fraction')

    if sample_fraction is not None:
        with h5py.File(output, 'r+') as f:
            f.attrs['sample_fraction'] = sample_fraction

    if source:
        log.info('Copying "runs" group')
        to_h5py(runs, output, key='runs', mode='a')


if __name__ == '__main__':
    main()
