import click
import yaml
import pandas as pd
from tqdm import tqdm
import h5py
import logging

from fact.io import read_data, write_data, h5py_get_n_rows
from ..apply import create_mask_h5py, apply_cuts_h5py_chunked, build_query


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
@click.option('-v', '--verbose', help='Verbose log output', type=bool)
def main(configuration_path, input_path, output_path, hdf_style, chunksize, key, verbose):
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

    selection = config.get('selection', {})

    if hdf_style == 'pandas':

        query = build_query(selection)

        log.info('Using query: ' + query)

        if chunksize is None:
            df = read_data(input_path, key=key)
            n_events = len(df)
            df = df.query(query)
            log.info('Before cuts: {}, after cuts: {}'.format(n_events, len(df)))
            df.to_hdf(output_path, key=key)
        else:
            with pd.HDFStore(output_path, mode='w') as store:
                it = pd.read_hdf(input_path, key=key, chunksize=chunksize)
                for df in tqdm(it):
                    store.append(key, df.query(query))

    else:
        if chunksize is None:

            n_events = h5py_get_n_rows(input_path, key=key)

            mask = create_mask_h5py(input_path, selection, key=key)
            log.info('Before cuts: {}, after cuts: {}'.format(n_events, mask.sum()))

            with h5py.File(input_path) as infile, h5py.File(output_path, 'w') as outfile:
                group = outfile.create_group(key)

                for name, dataset in infile[key].items():

                    if dataset.ndim == 1:
                        group.create_dataset(name, data=dataset[mask], maxshape=(None, ))
                    elif dataset.ndim == 2:
                        group.create_dataset(
                            name, data=dataset[mask, :], maxshape=(None, 2)
                        )
                    else:
                        log.warning('Skipping not 1d or 2d column {}'.format(name))
        else:
            apply_cuts_h5py_chunked(
                input_path, output_path, selection, chunksize=chunksize, key=key
            )

        with h5py.File(input_path) as infile, h5py.File(output_path, 'r+') as outfile:

            if 'runs' in infile.keys():
                log.info('Copying runs group to outputfile')
                infile.copy('/runs', outfile['/'])
