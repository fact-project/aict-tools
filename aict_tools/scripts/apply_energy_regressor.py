import click
from tqdm import tqdm

import pandas as pd

from ..apply import predict_energy
from ..io import (
    append_column_to_hdf5,
    read_telescope_data_chunked,
    drop_prediction_column,
    drop_prediction_groups,
    load_model,
)
from ..configuration import AICTConfig
from ..logging import setup_logging


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
    log = setup_logging(verbose=verbose)
    config = AICTConfig.from_yaml(configuration_path)
    model_config = config.energy
    prediction_column_name = model_config.output_name

    if config.data_format == 'CTA':
        drop_prediction_groups(
            data_path,
            group_name=model_config.output_name,
            yes=yes
        )
    elif config.data_format == 'simple':
        drop_prediction_column(
            data_path,
            group_name=config.events_key,
            column_name=model_config.output_name,
            yes=yes
        )

    log.debug('Loading model')
    model = load_model(model_path)
    log.debug('Done')

    if n_jobs:
        model.n_jobs = n_jobs

    df_generator = read_telescope_data_chunked(
        data_path, config, chunksize, model_config.columns_to_read_apply,
        feature_generation_config=model_config.feature_generation
    )

    if config.data_format == 'CTA':
        chunked_frames = []

    for df_data, start, stop in tqdm(df_generator):
        print(df_data.columns)
        print(model_config.features)

        energy_prediction = predict_energy(
            df_data[model_config.features],
            model,
            log_target=model_config.log_target,
        )

        if config.data_format == 'CTA':
            d = df_data[['obs_id', 'event_id']].copy()
            d[prediction_column_name] = energy_prediction
            chunked_frames.append(d)
            for tel in df_data['tel_id'].unique():
                table = f'/dl2/event/telescope/tel_{tel:03d}/{model_config.output_name}'
                matching = (df_data['tel_id'] == tel)
                d[matching].to_hdf(
                    data_path,
                    table,
                    mode='a',
                    format='table'
                )
        elif config.data_format == 'simple':
            append_column_to_hdf5(
                data_path,
                energy_prediction,
                config.events_key,
                model_config.output_name
            )

    if config.data_format == 'CTA':
        # Any reason to have this configurable?
        array_table = f'/dl2/event/subarray/{model_config.output_name}'
        d = pd.concat(chunked_frames).groupby(
            ['obs_id', 'event_id'], sort=False
        ).agg(['mean', 'std'])

        d.columns = d.columns.droplevel(0)
        d.reset_index().to_hdf(
            data_path,
            array_table,
            mode='a',
            format='table'
        )

if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
