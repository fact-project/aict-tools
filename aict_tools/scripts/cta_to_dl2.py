import click
from tqdm import tqdm
import pandas as pd
import os
import logging
from sklearn.externals import joblib

from fact.io import to_h5py

from aict_tools.apply import predict_energy, predict_separator
from aict_tools.io import read_telescope_data_chunked, read_data
from aict_tools.configuration import AICTConfig
from aict_tools.feature_generation import feature_generation


dl2_tel_event_columns = [
    'gamma_energy_prediction',
    'gamma_prediction',
    'pointing_altitude',
    'pointing_azimuth',
]

ids = ['telescope_id', 'array_event_id', 'run_id']

dl2_array_event_columns = [
    'pointing_altitude', 'pointing_azimuth',
    'altitude', 'azimuth', 'mc_az', 'mc_alt', 'az', 'alt', 'num_triggered_mst', 'num_triggered_sst',
    'num_triggered_lst', 'num_triggered_telescopes', 'mc_energy', 'mc_core_x', 'mc_core_y', 'mc_h_first_int', 'mc_x_max'
]


def flatten_multi_index(df):
    # thanks SO
    # https://stackoverflow.com/questions/27071661/pivoting-pandas-dataframe-into-prefixed-cols-not-a-multiindex
    mi = df.columns
    prefixes, suffixes = mi.levels
    col_names = [prefixes[i_p] + '_' + suffixes[i_s] for (i_p, i_s) in zip(*mi.labels)]
    df.columns = col_names
    return df



@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('data_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('separator_model_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('energy_model_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('output', type=click.Path(exists=False, dir_okay=False))
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
    output,
    chunksize,
    n_jobs,
    yes,
    verbose,
):
    '''
    Apply given models to data. 

    CONFIGURATION_PATH: Path to the config yaml file

    DATA_PATH: path to the CTA data in an hdf5 file, e.g. process_simtel_file output

    SEPARATOR_MODEL_PATH: Path to the pickled separation model.

    ENERGY_MODEL_PATH: Path to the pickled energy regression model.
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
        os.remove(output)

    log.info('Loading models')
    separator_model = joblib.load(separator_model_path)
    energy_model = joblib.load(energy_model_path)
    log.info('Done')

    if n_jobs:
        separator_model.n_jobs = n_jobs
        energy_model.n_jobs = n_jobs

    columns = set(dl2_tel_event_columns + ids)
    for model in ('separator', 'energy'):
        model_config = getattr(config, model)
        columns |= set(model_config.columns_to_read_apply)

    df_generator = read_telescope_data_chunked(
        data_path,
        config,
        chunksize=chunksize,
        columns=columns,
    )

    predictions = []
    aggregation_funcs = [min, max, 'mean', 'std', 'median']
    aggregation = {
        'gamma_prediction': aggregation_funcs, 
        'gamma_energy_prediction': aggregation_funcs,
    }

    log.info('Predicting on data...')
    for df, _, _ in tqdm(df_generator):
        df_sep = feature_generation(df, config.separator.feature_generation)
        df['gamma_prediction'] = predict_separator(
            df_sep[config.separator.features], separator_model,
        )
        assert len(df) == len(df_sep)
        df_energy = feature_generation(df, config.energy.feature_generation)
        df['gamma_energy_prediction'] = predict_energy(
            df_energy[config.energy.features],
            energy_model,
            log_target=config.energy.log_target,
        )
        assert len(df) == len(df_energy)
                
        gdf = df.groupby(['run_id', 'array_event_id']).agg(aggregation)
        flatten_multi_index(gdf)
        predictions.append(gdf)


    predictions = pd.concat(predictions)
    df = read_data(data_path, key='array_events', columns=dl2_array_event_columns + ['array_event_id', 'run_id'])
    df_merged = pd.merge(predictions, df, left_index=True, right_on=['run_id', 'array_event_id'])
    
    to_h5py(df_merged, output, key='array_events', mode='a')
    
    runs = read_data(data_path, key='runs')
    to_h5py(runs, output, key='runs', mode='a')


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
