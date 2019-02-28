import numpy as np
import click
import yaml
import h5py
import logging
from tqdm import tqdm 
import pandas as pd
from ..io import get_number_of_rows_in_table, read_data_chunked, read_data, write_hdf
from ..apply import apply_cuts_h5py_chunked #, create_mask_dataframe, create_mask_h5py
from colorama import Fore, Style


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('input_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_path', type=click.Path(exists=False, dir_okay=False))
@click.option('-N', '--chunksize', type=int, help='Chunksize to use')
@click.option('-k', '--key', help='Name of the hdf5 group', default=None)
@click.option('-v', '--verbose', help='Verbose log output', is_flag=True)
def main(configuration_path, input_path, output_path, chunksize, key, verbose):
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
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    log = logging.getLogger()

    with open(configuration_path) as f:
        config = yaml.safe_load(f)
    multiple_telescopes = config['multiple_telescopes']
    selection = config.get('selection', {})
    if not multiple_telescopes and not key:
        key = 'events'
    if multiple_telescopes and not key:
        key = 'telescope_events'
    
    n_events = get_number_of_rows_in_table(input_path, key=key)
    print(f'Events in file before cuts {n_events}')
    if chunksize is None:
        chunksize = n_events + 1

    apply_cuts_h5py_chunked(
        input_path, output_path, selection, chunksize=chunksize, key=key
    )
    if multiple_telescopes:
        print('Copying selected array events.')
        # read incdex of remaining telescope events.
        df_index = read_data(output_path, key='telescope_events', columns=['array_event_id', 'run_id', 'telescope_id'])
        df_index.set_index(['run_id', 'array_event_id',], inplace=True)
        df_index = df_index[~df_index.index.duplicated()]
        
        array_events = read_data(input_path, 'array_events')
        array_events.set_index(['run_id', 'array_event_id'], inplace=True)
        array_events['index_in_file'] = np.arange(0, len(array_events)) 
        
        array_events = pd.merge(array_events, df_index, left_index=True, right_index=True, validate='one_to_one')
        array_events.sort_values('index_in_file', inplace=True) 
        write_hdf(array_events, output_path, table_name='array_events', mode='a')
    

    with h5py.File(input_path, mode='r') as infile, h5py.File(output_path, 'r+') as outfile:
        if 'runs' in infile.keys():
            log.info('Copying runs group to outputfile')
            infile.copy('/runs', outfile['/'])

    n_events_after = get_number_of_rows_in_table(output_path, key=key)
    percentage = 100 * n_events_after/n_events
    print(f'Events in new file after cuts {n_events_after}. That is {percentage:.2f} %')
