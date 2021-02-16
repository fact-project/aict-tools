import click
from ruamel.yaml import YAML
from shutil import copyfile

from ..io import (
    get_number_of_rows_in_table,
    copy_group,
)
from ..apply import apply_cuts_h5py_chunked, apply_cuts_cta_dl1
from ..logging import setup_logging

yaml = YAML(typ='safe')


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('input_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_path', type=click.Path(exists=False, dir_okay=False))
@click.option(
    '-N',
    '--chunksize',
    type=int,
    help='How many events to read at once, only supported for single telescope h5py files.')
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

    selection = config.get('selection', None)
    data_format = config.get('data_format', 'simple')

    if not selection:
        log.info('No entries for selection cuts. Just copying files.')
        copyfile(input_path, output_path)
        log.info('Copying finished')
        return

    log.info(data_format)
    if data_format == 'simple':
        key = config.get('events_key', 'events')
        n_events = get_number_of_rows_in_table(input_path, key=key)
        if chunksize is None:
            chunksize = n_events + 1
        apply_cuts_h5py_chunked(
            input_path,
            output_path,
            selection,
            chunksize=chunksize,
            key=key
        )
        n_events_after = get_number_of_rows_in_table(output_path, key=key)
        remaining = n_events_after / n_events
        log.info(f'Events in file before cuts {n_events}')
        log.info(f'Events in new file after cuts {n_events_after}. That is {remaining:.2%}')
        copy_group(input_path, output_path, 'runs')
    elif data_format == 'CTA':
        keep_images = config.get('keep_images', True)
        n_before, n_after = apply_cuts_cta_dl1(
            input_path,
            output_path,
            selection,
            keep_images,
        )

        log.info(f'Telescope-events in file before cuts {n_before}')
        log.info(
            f'Telescope-events in new file after cuts {n_after}. '
            f'That is {(n_after/n_before):.2%}'
        )
