import numpy as np
import click
from ruamel.yaml import YAML
from tqdm import tqdm
import pandas as pd
from shutil import copyfile

from ..io import (
    get_number_of_rows_in_table,
    read_data_chunked,
    read_data,
    write_hdf,
    copy_runs_group,
)
from ..apply import apply_cuts_h5py_chunked
from ..logging import setup_logging

yaml = YAML(typ='safe')


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('input_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_path', type=click.Path(exists=False, dir_okay=False))
@click.option('-N', '--chunksize', type=int, help='Chunksize to use')
@click.option('-v', '--verbose', help='Verbose log output', is_flag=True)
def main(configuration_path, input_path, output_path, chunksize, verbose):
    '''
    Apply cuts given in CONFIGURATION_PATH to the data in INPUT_PATH and
    write the result to OUTPUT_PATH.

    example:
    ```
    selection:
        numPixelInShower: ['>=', 10]
        numIslands: ['<=', 5]
        Width: ['<=', 50]
    ```
    '''
    log = setup_logging(verbose=verbose)

    with open(configuration_path) as f:
        config = yaml.load(f)

    multiple_telescopes = config.get('multiple_telescopes', False)
    selection = config.get('selection', None)

    if multiple_telescopes:
        key = 'telescope_events'
    else:
        key = 'events'

    if not selection:
        log.info('No entries for selection cuts. Just copying files.')
        copyfile(input_path, output_path)
        log.info('Copying finished')
        return

    n_events = get_number_of_rows_in_table(input_path, key=key)
    if chunksize is None:
        chunksize = n_events + 1

    apply_cuts_h5py_chunked(
        input_path, output_path, selection, chunksize=chunksize, key=key
    )
    if multiple_telescopes:
        log.info('Copying selected array events.')
        # read incdex of remaining telescope events.
        df_index = read_data(
            output_path,
            key='telescope_events',
            columns=['array_event_id', 'run_id', 'telescope_id']
        )
        df_index.set_index(['run_id', 'array_event_id'], inplace=True)
        df_index = df_index[~df_index.index.duplicated()]

        df_iterator = read_data_chunked(input_path, 'array_events', chunksize=500000)
        for array_events, _, _ in tqdm(df_iterator):
            array_events.set_index(['run_id', 'array_event_id'], inplace=True)
            array_events['index_in_file'] = np.arange(0, len(array_events))

            array_events = pd.merge(
                array_events,
                df_index,
                left_index=True,
                right_index=True,
                validate='one_to_one',
            )
            array_events.sort_values('index_in_file', inplace=True)
            array_events.drop('index_in_file', axis='columns', inplace=True,)
            if len(array_events > 0):
                write_hdf(array_events, output_path, table_name='array_events', mode='a')

    copy_runs_group(input_path, output_path)

    n_events_after = get_number_of_rows_in_table(output_path, key=key)
    remaining = n_events_after / n_events
    log.info(f'Events in file before cuts {n_events}')
    log.info(f'Events in new file after cuts {n_events_after}. That is {remaining:.2%}')
