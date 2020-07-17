import numpy as np
import click
from ruamel.yaml import YAML
from shutil import copyfile

from ..io import (
    get_number_of_rows_in_table,
    copy_group,
)
from ..apply import apply_cuts_h5py_chunked, create_mask_h5py
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
    # ToDo: There is a bug with chunksize applying -> Numbers change depending on chunksize
    # ToDo: Remove events with no more telescopes (+verify this doesnt break the file right now)
    elif data_format == 'CTA':
        import tables
        # copy the tables we do not perform cuts on
        for key in [
                'simulation',
                'configuration',
                'dl1/service',
                'dl1/monitoring',
                'dl1/event/subarray',
                'dl1/event/telescope/trigger',
                'dl2'
        ]:
            copy_group(input_path, output_path, key)

        n_total_before = 0
        n_total_after = 0
        infile = tables.open_file(input_path)
        outfile = tables.open_file(output_path, 'a')
        outfile.create_group('/dl1/event/telescope', 'parameters')
        for tel in infile.root.dl1.event.telescope.parameters:
            key = f'/dl1/event/telescope/parameters/{tel.name}'
            in_rows = tel.iterrows()
            desc = tel.description._v_colobjects.copy()
            out_table = outfile.create_table(
                '/dl1/event/telescope/parameters',
                tel.name,
                desc)

            n_events = get_number_of_rows_in_table(input_path, key=key)
            if chunksize is None:
                tel_chunksize = n_events + 1
            else:
                tel_chunksize = chunksize
            n_chunks = int(np.ceil(n_events/tel_chunksize))

            for chunk in range(n_chunks):
                start = chunk * tel_chunksize
                end = min(n_events, (chunk+1) * tel_chunksize)
                mask = create_mask_h5py(
                    infile,
                    selection,
                    key=key,
                    n_events=n_events,
                    start=start,
                    end=end,
                )
                for row, match in zip(in_rows, mask):
                    if match:
                        out_table.append([row[:]])

            n_events_after = get_number_of_rows_in_table(output_path, key=key)
            n_total_before += n_events
            n_total_after += n_events_after

        log.info(f'Events in file before cuts {n_total_before}')
        log.info(f'Events in new file after cuts {n_total_after}. That is {(n_total_after/n_total_before):.2%}')
