import click
import yaml
import pandas as pd
from tqdm import tqdm
import h5py
import numpy as np
import operator as o
import logging

from ..io import read_pandas_hdf5, write_data, h5py_get_n_events

operators = {
    '<': o.lt, 'lt': o.lt,
    '<=': o.le, 'le': o.le,
    '==': o.eq, 'eq': o.eq,
    '=': o.eq,
    '!=': o.ne, 'ne': o.ne,
    '>=': o.ge, 'ge': o.ge,
    '>': o.gt, 'gt': o.gt
}


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('input_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_path', type=click.Path(exists=False, dir_okay=False))
@click.option(
    '-h', '--hdf-style', default='pandas', type=click.Choice(['pandas', 'h5py']),
    help='Wether to use pandas or h5py for the output file'
)
@click.option('-N', '--chunksize', type=int, help='Chunksize to use')
@click.option('-k', '--key', help='Name of the hdf5 group', default='table')
def main(configuration_path, input_path, output_path, hdf_style, chunksize, key):
    '''
    Apply cuts given in CONFIGURATION_PATH to the data in INPUT_PATH and
    write the result to OUTPUT_PATH.

    example:
    ```
    selection:
        numPixelInShower: ['>=', 10']
        numIslands: ['<=', 5]
        Width: ['<=', 50]
    ```
    '''
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger()

    with open(configuration_path) as f:
        config = yaml.safe_load(f)

    selection = config.get('selection', {})

    if hdf_style == 'pandas':

        queries = ('{} {} {}'.format(k, o, '"' + v + '"' if isinstance(v, str) else v)
                   for k, (o, v) in selection.items())

        query = '(' + ') & ('.join(queries) + ')'
        log.info('Using query: ' + query)

        if chunksize is None:
            df = read_pandas_hdf5(input_path, key=key)
            n_events  = len(df)
            df = df.query(query)
            log.info('Before cuts: {}, after cuts: {}'.format(n_events, len(df)))
            write_data(df, output_path, hdf_key=key)
        else:
            with pd.HDFStore(output_path, mode='w') as store:
                it = read_pandas_hdf5(input_path, key=key, chunksize=chunksize)
                for df in tqdm(it):
                    store.append(key, df.query(query))

    else:
        if chunksize is None:
            n_events = h5py_get_n_events(input_path, key=key)
            mask = np.ones(n_events, dtype=bool)

            with h5py.File(input_path) as infile, h5py.File(output_path, 'w') as outfile:

                for name, (operator, value) in selection.items():

                    mask = np.logical_and(
                        mask, operators[operator](infile[key][name][:], value)
                    )

                log.info('Before cuts: {}, after cuts: {}'.format(n_events, mask.sum()))

                group = outfile.create_group(key)

                for name, dataset in infile[key].items():

                    if dataset.ndim == 1:
                        group.create_dataset(name, data=dataset[mask])
                    else:
                        log.warning('Skipping not 1d column {}'.format(name))
        else:
            raise NotImplementedError('Chunking not yet implemented for h5py')
