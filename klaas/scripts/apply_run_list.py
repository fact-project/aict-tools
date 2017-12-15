import click
import yaml
import pandas as pd
from tqdm import tqdm
import h5py
import logging

from fact.io import read_data, h5py_get_n_rows
from ..apply import create_run_list_mask_h5py, apply_run_list_h5py_chunked, build_query




@click.command()
@click.argument('run_list_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('input_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_path', type=click.Path(exists=False, dir_okay=False))
@click.option(
    '-h', '--hdf-style', default='h5py', type=click.Choice(['pandas', 'h5py']),
    help='Wether to use pandas or h5py for the output file'
)
@click.option('-N', '--chunksize', type=int, help='Chunksize to use')
@click.option('-k', '--key', help='Name of the hdf5 group', default='events')
@click.option('-m', '--mode', help='Excess mode of the input file', default='r')
@click.option('-v', '--verbose', help='Verbose log output', is_flag=True)
def main(run_list_path, input_path, output_path, hdf_style, chunksize, key, mode, verbose):
    '''
     Apply a given list of runs from a list (CSV) in RUN_LIST_PATH to the data
     in INPUT_PATH an write the result to OUTPUT_PATH.
    '''
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    log = logging.getLogger()
    
    if hdf_style == 'pandas':    
        print("pandas!")
        query = build_run_list_query(run_list_path)
        
        log.info('Using query: ' + query)
        
        runs = read_data(input_path, key='runs', mode=mode)
        runs = runs.query(query)
        runs.to_hdf(output_path, key='runs')
    
        if chunksize is None:
            print("No chunks")
            df = read_data(input_path, key=key, mode=mode)
            
            n_events = len(df)
            df = df.query(query)
            
            log.info('Before cuts: {}, after cuts: {}'.format(n_events, len(df)))
            df.to_hdf(output_path, key=key)
            runs.to_hdf(output_path, key='runs')
        else:
            with pd.HDFStore(output_path, mode=mode) as store:
                it = pd.read_hdf(input_path, key=key, chunksize=chunksize)
                for df in tqdm(it):
                    store.append(key, df.query(query))
    
    else:
        print("h5py!")
        if chunksize is None:
            print("No chunks")
            apply_runlist_to_data_set(input_path, output_path, key, run_list_path)
            apply_runlist_to_data_set(input_path, output_path, "runs", run_list_path)
            
        else:
            apply_run_list_h5py_chunked(
                input_path, output_path, run_list_path, chunksize=chunksize, key=key
            )
            
            apply_run_list_h5py_chunked(
                input_path, output_path, run_list_path, chunksize=chunksize, key="runs"
            )


def apply_runlist_to_data_set(input_path, output_path, key, run_list_path):
    n_events = h5py_get_n_rows(input_path, key=key, mode=mode)

    mask = create_run_list_mask_h5py(input_path, run_list_path, key=key)
    log.info('Before cuts: {}, after cuts: {}'.format(n_events, mask.sum()))

    with h5py.File(input_path) as infile, h5py.File(output_path, 'w') as outfile:
        group = outfile.create_group(key)
        print("banana!")
        for name, dataset in infile[key].items():

            if dataset.ndim == 1:
                group.create_dataset(name, data=dataset[mask], maxshape=(None, ))
            elif dataset.ndim == 2:
                group.create_dataset(
                    name, data=dataset[mask, :], maxshape=(None, 2)
                )
            else:
                log.warning('Skipping not 1d or 2d column {}'.format(name))
                
        for name, dataset in infile["runs"].items():

            if dataset.ndim == 1:
                runs_group.create_dataset(name, data=dataset[runs_mask], maxshape=(None, ))
            elif dataset.ndim == 2:
                runs_group.create_dataset(
                    name, data=dataset[runs_mask, :], maxshape=(None, 2)
                )
            else:
                log.warning('Skipping not 1d or 2d column {}'.format(name))
                
