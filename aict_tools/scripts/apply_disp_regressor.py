import click
import numpy as np
from sklearn.externals import joblib
import logging
import h5py
from tqdm import tqdm

from ..io import append_to_h5py, read_telescope_data_chunked
from ..apply import predict_disp
from ..configuration import AICTConfig


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('data_path', type=click.Path(exists=True, dir_okay=False))
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

    config = AICTConfig.from_yaml(configuration_path)
    model_config = config.disp

    columns_to_delete = [
        'source_x_prediction',
        'source_y_prediction',
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

    n_del_cols = 0
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
                log.warn("Deleted {} from the feature set.".format(column))
                n_del_cols += 1

    if n_del_cols > 0:
        log.warn("Source dependent features need to be calculated from the predicted source possition. "
                 + "Use e.g. `fact_calculate_theta` from https://github.com/fact-project/pyfact.")

    log.info('Loading model')
    disp_model = joblib.load(disp_model_path)
    sign_model = joblib.load(sign_model_path)
    log.info('Done')

    if n_jobs:
        disp_model.n_jobs = n_jobs
        sign_model.n_jobs = n_jobs

    df_generator = read_telescope_data_chunked(
        data_path, config, chunksize, model_config.columns_to_read_apply,
        feature_generation_config=model_config.feature_generation
    )

    log.info('Predicting on data...')
    for df_data, start, end in tqdm(df_generator):

        disp = predict_disp(
            df_data[model_config.features], disp_model, sign_model
        )

        source_x = df_data.cog_x + disp * np.cos(df_data.delta)
        source_y = df_data.cog_y + disp * np.sin(df_data.delta)

        with h5py.File(data_path, 'r+') as f:
            append_to_h5py(f, source_x, key, 'source_x_prediction')
            append_to_h5py(f, source_y, key, 'source_y_prediction')
            append_to_h5py(f, disp, key, 'disp_prediction')


if __name__ == '__main__':
    main()
