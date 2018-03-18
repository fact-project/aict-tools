import click
import yaml
import h5py
import logging

from fact.io import h5py_get_n_rows
from ..apply import create_mask_h5py, apply_cuts_h5py_chunked


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('input_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_path', type=click.Path(exists=False, dir_okay=False))
@click.option('-N', '--chunksize', type=int, help='Chunksize to use')
@click.option('-k', '--key', help='Name of the hdf5 group', default='events')
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

    selection = config.get('selection', {})

    if chunksize is None:
        n_events = h5py_get_n_rows(input_path, key=key, mode='r')

        mask = create_mask_h5py(input_path, selection, key=key)
        log.info('Before cuts: {}, after cuts: {}'.format(n_events, mask.sum()))

        with h5py.File(input_path, mode='r') as infile, h5py.File(output_path, 'w') as outfile:
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

    with h5py.File(input_path, mode='r') as infile, h5py.File(output_path, 'r+') as outfile:
        if 'runs' in infile.keys():
            log.info('Copying runs group to outputfile')
            infile.copy('/runs', outfile['/'])
