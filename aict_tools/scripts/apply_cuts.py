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

    # if multiple_telescopes:
    #     print('Copying telescope events.')
    #     # this guarantees that array events are bunched together in here.
    #     df_iterator = read_data_chunked(input_path, 'telescope_events', chunksize=chunksize) 
    #     for telescope_events, _, _ in tqdm(df_iterator):
    #         mask = create_mask_dataframe(telescope_events, selection)
    #         telescope_events = telescope_events[mask]

    #         # create mask to select all duplicate index entries. we want to keep those 
    #         # and remove remaining single telescope events.
    #         telescope_events.set_index(['run_id', 'array_event_id'], inplace=True)
    #         selected = telescope_events.index.duplicated(keep=False)
    #         telescope_events = telescope_events[selected]
    #         write_hdf(telescope_events, output_path, table_name='telescope_events', mode='a')

    #     print('Copying selected array events.')
    #     # read incdex of remaining telescope events.
    #     df_index = read_data(output_path, key='telescope_events', columns=['array_event_id', 'run_id', 'telescope_id'])
    #     df_index.set_index(['run_id', 'array_event_id',], inplace=True)
    #     df_index = df_index[~df_index.index.duplicated()]

    #     df_iterator = read_data_chunked(input_path, 'array_events', chunksize=chunksize//5)
    #     for array_events, _, _ in tqdm(df_iterator):
    #         array_events.set_index(['run_id', 'array_event_id'], inplace=True)
    #         print(f'Have  {len(array_events)} array events')
    #         array_events = pd.merge(array_events, df_index, left_index=True, right_index=True, validate='one_to_one')
    #         print(f'writing  {len(array_events)} to file')
    #         write_hdf(array_events, output_path, table_name='array_events', mode='a')
    
    # else:
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

def verify_file(input_file_path):
    try:
        
        telescope_events = read_data(input_file_path, key='telescope_events')
        array_events = read_data(input_file_path, key='array_events')
        runs = read_data(input_file_path, key='runs')

        runs.set_index('run_id', drop=True, verify_integrity=True, inplace=True)
        telescope_events.set_index(['run_id', 'array_event_id', 'telescope_id'], drop=True, verify_integrity=True, inplace=True)
        array_events.set_index(['run_id', 'array_event_id'], drop=True, verify_integrity=True, inplace=True)
        result = pd.merge(runs, array_events, left_index=True, right_index=True, validate='one_to_many')  
        assert result.shape[0] == array_events.shape[0]

        telescope_events.reset_index(drop=False, inplace=True)
        telescope_events.set_index(['run_id', 'array_event_id'], inplace=True)
        result = pd.merge(array_events, telescope_events, left_index=True, right_index=True, validate='one_to_many')
        assert result.shape[0] == telescope_events.shape[0]

        print(Fore.GREEN + Style.BRIGHT + f'File "{input_file_path}" seems fine.   ✔ ')
        print(Style.RESET_ALL)   
    except:
        print(Fore.RED + f'File {input_file_path} seems to be broken. \n')
        print(Style.RESET_ALL)   
        import sys, traceback
        traceback.print_exc(file=sys.stdout)