import click
import numpy as np
from sklearn.externals import joblib
import logging
import h5py
from tqdm import tqdm
from fact.coordinates.utils import camera_to_horizontal
from ..cta_coordinates import camera_to_horizontal as camera_to_horizontal_cta
from ..cta_coordinates import horizontal_to_camera as horizontal_to_camera_cta
from ..io import append_to_h5py, read_telescope_data_chunked
from ..apply import predict_disp
from ..configuration import AICTConfig
from ..preprocessing import calc_true_disp
import astropy.units as u
from sklearn.metrics import accuracy_score, r2_score


log = logging.getLogger()


def print_metrics(disp_prediction, true_disp, true_sign):
    valid = ~np.isnan(disp_prediction)
    acc = accuracy_score(np.sign(true_sign[valid]), np.sign(disp_prediction[valid]))
    print(f'Accuracy sign classififier: {acc}')

    r2 = r2_score((true_disp * true_sign)[valid], disp_prediction[valid])
    print(f'R2 score regressor: {r2}')

    r2 = r2_score(true_disp[valid], np.abs(disp_prediction[valid]))
    print(f'R2 score abs(regressor): {r2}')


def delete_old_prediction_columns(data_path, config, yes):
    columns_to_delete = [
        'source_x_prediction',
        'source_y_prediction',
        'az_prediction',
        'alt_prediction',
        'zd_prediction',
        'az_prediction_disp',
        'alt_prediction_disp',
        'zd_prediction_disp',
        'az_prediction_degree',
        'alt_prediction_degree',
        'zd_prediction_degree',
        'theta',
        'theta_deg',
        'theta_rec_pos',
        'disp_prediction',
        'mc_disp',
    ]
    for i in range(1, 6):
        columns_to_delete.extend([
            'theta_off_' + str(i),
            'theta_deg_off_' + str(i),
            'theta_off_rec_pos_' + str(i),
        ])

    n_del_cols = 0
    with h5py.File(data_path, 'r+') as f:
        for column in columns_to_delete:
            if column in f[config.telescope_events_key].keys():
                if not yes:
                    click.confirm(
                        'Dataset "{}" exists in file, overwrite?'.format(column),
                        abort=True,
                    )
                    yes = True
                del f[config.telescope_events_key][column]
                log.warn("Deleted {} from the feature set.".format(column))
                n_del_cols += 1

    if n_del_cols > 0:
        log.warn("Source dependent features need to be calculated from the predicted source possition. "
                 + "Use e.g. `fact_calculate_theta` from https://github.com/fact-project/pyfact.")


def calculate_true_source(df, model_config):
        az = df[model_config.source_azimuth_column].values * u.rad
        alt = df[model_config.source_altitude_column].values * u.rad
        az_pointing = df[model_config.pointing_azimuth_column].values * u.rad
        alt_pointing = df[model_config.pointing_altitude_column].values * u.rad
        focal_length = df[model_config.focal_length_column].values * u.m

        source_x, source_y = horizontal_to_camera_cta(
            az=az,
            alt=alt,
            az_pointing=az_pointing,
            alt_pointing=alt_pointing,
            focal_length=focal_length,
        )

        source_x = source_x.to(u.m).value
        source_y = source_y.to(u.m).value

        return source_x, source_y


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('data_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('disp_model_path', type=click.Path(exists=False, dir_okay=False))
@click.argument('sign_model_path', type=click.Path(exists=False, dir_okay=False))
@click.option('-n', '--n-jobs', type=int, help='Number of cores to use')
@click.option('-y', '--yes', help='Do not prompt for overwrites', is_flag=True)
@click.option('-v', '--verbose', help='Verbose log output', is_flag=True)
@click.option(
    '-N', '--chunksize', type=int,
    help='If given, only process the given number of events at once',
)
def main(configuration_path, data_path, disp_model_path, sign_model_path, chunksize, n_jobs, yes, verbose):
    '''
    Apply given model to data. Two columns are added to the file, energy_prediction
    and energy_prediction_std

    CONFIGURATION_PATH: Path to the config yaml file
    DATA_PATH: path to the FACT data in a h5py hdf5 file, e.g. erna_gather_fits output
    DISP_MODEL_PATH: Path to the pickled disp model.
    SIGN_MODEL_PATH: Path to the pickled sign model.
    '''
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    config = AICTConfig.from_yaml(configuration_path)

    delete_old_prediction_columns(data_path, config, yes)

    log.info('Loading model')
    disp_model = joblib.load(disp_model_path)
    sign_model = joblib.load(sign_model_path)
    log.info('Done')

    if n_jobs:
        disp_model.n_jobs = n_jobs
        sign_model.n_jobs = n_jobs

    model_config = config.disp
    df_generator = read_telescope_data_chunked(
        data_path, config, chunksize, model_config.columns_to_read_apply,
        feature_generation_config=model_config.feature_generation
    )

    log.info('Predicting on data...')
    for df_data, start, end in tqdm(df_generator):

        disp_prediction = predict_disp(
            df_data[model_config.features], disp_model, sign_model
        )
        delta = df_data[model_config.delta_column].values
        if config.has_multiple_telescopes:
            delta = np.where(delta > np.pi/2, delta - np.pi, delta)
            delta = np.where(delta < -np.pi/2, delta + np.pi, delta)

        cog_x = df_data[model_config.cog_x_column]
        cog_y = df_data[model_config.cog_y_column]
        source_x_prediction = cog_x + disp_prediction * np.cos(delta)
        source_y_prediction = cog_y + disp_prediction * np.sin(delta)

        source_x, source_y = calculate_true_source(df_data, model_config)
        true_disp, true_sign = calc_true_disp(
            source_x, source_y,
            cog_x, cog_y,
            delta,
        )

        if config.has_multiple_telescopes:
            # use cta units. meters and radians
            az_pointing = df_data[model_config.pointing_azimuth_column].values * u.rad
            alt_pointing = df_data[model_config.pointing_altitude_column].values * u.rad
            focal_length = df_data[model_config.focal_length_column].values * u.m

            altitude_prediction, azimuth_prediction = camera_to_horizontal_cta(
                x=source_x_prediction.values * u.m,
                y=source_y_prediction.values * u.m,
                az_pointing=az_pointing,
                alt_pointing=alt_pointing,
                focal_length=focal_length,
            )

            azimuth_prediction = np.where(
                azimuth_prediction.to(u.rad) > np.pi*u.rad,
                (azimuth_prediction.to(u.rad) - 2*np.pi*u.rad).to(u.rad),
                azimuth_prediction.to(u.rad)
            ) * u.rad
            zenith_prediction = 90 * u.deg - altitude_prediction

        else:
            zenith_prediction, azimuth_prediction = camera_to_horizontal(
                y=source_x_prediction,
                x=source_y_prediction,
                az_pointing=df_data[model_config.pointing_azimuth_column],
                zd_pointing=df_data[model_config.pointing_zenith_column],
            )
            altitude_prediction = (90 - zenith_prediction) * u.deg
            zenith_prediction = zenith_prediction * u.deg
            azimuth_prediction = azimuth_prediction * u.deg


        with h5py.File(data_path, 'r+') as f:
            k = config.telescope_events_key
            append_to_h5py(f, source_x_prediction, k, 'source_x_prediction')
            append_to_h5py(f, source_y_prediction, k, 'source_y_prediction')

            append_to_h5py(f, zenith_prediction.to(u.deg).value, k, 'zd_prediction_degree')
            append_to_h5py(f, altitude_prediction.to(u.deg).value, k, 'alt_prediction_degree')
            append_to_h5py(f, azimuth_prediction.to(u.deg).value, k, 'az_prediction_degree')

            if config.has_multiple_telescopes:
                append_to_h5py(f, zenith_prediction.to(u.rad).value, k, 'zd_prediction_disp')
                append_to_h5py(f, altitude_prediction.to(u.rad).value, k, 'alt_prediction_disp')
                append_to_h5py(f, azimuth_prediction.to(u.rad).value, k, 'az_prediction_disp')
            else:
                append_to_h5py(f, zenith_prediction.to(u.rad).value, k, 'zd_prediction')
                append_to_h5py(f, altitude_prediction.to(u.rad).value, k, 'alt_prediction')
                append_to_h5py(f, azimuth_prediction.to(u.rad).value, k, 'az_prediction')

            append_to_h5py(f, disp_prediction, k, 'disp_prediction')
            append_to_h5py(f, true_disp * true_sign, k, 'mc_disp')


if __name__ == '__main__':
    main()
