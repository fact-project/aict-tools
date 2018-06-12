import click
from sklearn.externals import joblib
import logging
import h5py
from tqdm import tqdm

import pandas as pd

from ..apply import predict_energy
from ..io import append_to_h5py, read_telescope_data_chunked, drop_prediction_column
from ..configuration import AICTConfig


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('data_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('model_path', type=click.Path(exists=True, dir_okay=False))
@click.option('-n', '--n-jobs', type=int, help='Number of cores to use')
@click.option('-y', '--yes', help='Do not prompt for overwrites', is_flag=True)
@click.option('-v', '--verbose', help='Verbose log output', is_flag=True)
@click.option(
    '-N', '--chunksize', type=int,
    help='If given, only process the given number of events at once',
)
def main(configuration_path, data_path, model_path, chunksize, n_jobs, yes, verbose):
    '''
    Apply given model to data. Two columns are added to the file, energy_prediction
    and energy_prediction_std

    CONFIGURATION_PATH: Path to the config yaml file
    DATA_PATH: path to the FACT/CTA data in a h5py hdf5 file, e.g. erna_gather_fits output
    MODEL_PATH: Path to the pickled model
    '''
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    log = logging.getLogger()
    config = AICTConfig.from_yaml(configuration_path)
    model_config = config.energy

    prediction_column_name = config.class_name + '_energy_prediction'
    drop_prediction_column(
        data_path, group_name=config.telescope_events_key,
        column_name=prediction_column_name, yes=yes
    )
    if config.has_multiple_telescopes:
        drop_prediction_column(
            data_path, group_name=config.array_events_key,
            column_name=prediction_column_name, yes=yes
        )

    log.debug('Loading model')
    model = joblib.load(model_path)
    log.debug('Done')

    if n_jobs:
        model.n_jobs = n_jobs

    df_generator = read_telescope_data_chunked(
        data_path, config, chunksize, model_config.columns_to_read_apply,
        feature_generation_config=model_config.feature_generation
    )

    if config.has_multiple_telescopes:
        chunked_frames = []

    for df_data, start, end in tqdm(df_generator):

        energy_prediction = predict_energy(
            df_data[model_config.features],
            model,
            log_target=model_config.log_target,
        )

        if config.has_multiple_telescopes:
            d = df_data[['run_id', 'array_event_id']].copy()
            d[prediction_column_name] = energy_prediction
            chunked_frames.append(d)

        with h5py.File(data_path, 'r+') as f:
            append_to_h5py(
                f, energy_prediction, config.telescope_events_key, prediction_column_name
            )

    if config.has_multiple_telescopes:
        d = pd.concat(chunked_frames).groupby(
            ['run_id', 'array_event_id'], sort=False
        ).agg(['mean', 'std'])

        mean = d[prediction_column_name]['mean'].values
        std = d[prediction_column_name]['std'].values
        with h5py.File(data_path, 'r+') as f:
            append_to_h5py(
                f, mean, config.array_events_key, prediction_column_name + '_mean'
            )
            append_to_h5py(
                f, std, config.array_events_key, prediction_column_name + '_std'
            )


if __name__ == '__main__':
    main()
