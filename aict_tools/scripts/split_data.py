import click
import numpy as np
import logging

from fact.io import read_data, write_data, read_h5py
import warnings
from math import ceil
import h5py
from tqdm import tqdm
import h5py

log = logging.getLogger()


def split_indices(idx, n_total, fractions):
    '''
    splits idx containing n_total distinct events into fractions given in fractions list.
    returns the number of events in each split
    '''
    num_ids = [ceil(n_total * f) for f in fractions]
    if sum(num_ids) > n_total:
        num_ids[-1] -= sum(num_ids) - n_total
    return num_ids


@click.command()
@click.argument('input_path', type=click.Path(exists=True, file_okay=True))
@click.argument('output_basename')
@click.option(
    '--fraction', '-f', multiple=True, type=float,
    help='Fraction of events to use for this part'
)
@click.option(
    '--name',
    '-n',
    multiple=True,
    help='name for one dataset'
)
@click.option(
    '-i',
    '--inkey',
    help='HDF5 key for h5py hdf5 of the input file',
    default='events', show_default=True,
)
@click.option(
    '--key',
    '-k',
    help='Name for the hdf5 group in the output',
    default='events',
    show_default=True,
)
@click.option(
    '--telescope', '-t', type=click.Choice(['fact', 'cta']), default='fact',
    show_default=True, help='Which telescope created the data',
)
@click.option(
    '--fmt', type=click.Choice(['csv', 'hdf5', 'hdf', 'h5']), default='hdf5',
    show_default=True, help='The output format',
)
@click.option(
    '--chunksize',
    type=int,
    help='How many events to read at once, only supported for h5py and single telescopes'
)
@click.option('-s', '--seed', help='Random Seed', type=int, default=0, show_default=True)
@click.option('-v', '--verbose', is_flag=True, help='Verbose log output',)
def main(input_path, output_basename, fraction, name, inkey, key, telescope, fmt, chunksize, seed, verbose):
    '''
    Split dataset in INPUT_PATH into multiple parts for given fractions and names
    Outputs hdf5 or csv files to OUTPUT_BASENAME_NAME.FORMAT

    Example call: aict_split_data input.hdf5 output_base -n test -f 0.5 -n train -f 0.5
    '''

    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    log.debug("input_path: {}".format(input_path))

    assert len(fraction) == len(name), 'You must give a name for each fraction'

    if sum(fraction) != 1:
        warnings.warn('Fractions do not sum up to 1')

    np.random.seed(seed)

    if telescope == 'fact':
        if chunksize is None:
            split_single_telescope_data(input_path, output_basename, fmt, inkey, key, fraction, name)
        else:
            split_single_telescope_data_chunked(input_path, output_basename, inkey, key, fraction, name, chunksize)
    else:
        split_multi_telescope_data(input_path, output_basename, fraction, name)


def split_multi_telescope_data(input_path, output_basename, fraction, name):

    array_events = read_data(input_path, key='array_events')
    telescope_events = read_data(input_path, key='telescope_events')
    runs = read_data(input_path, key='runs')

    # split by runs

    ids = set(runs.run_id)
    log.debug(f'All runs:{ids}')
    n_total = len(runs)

    log.info(f'Found a total of {n_total} runs in the file')
    num_runs = split_indices(ids, n_total, fractions=fraction)

    for n, part_name in zip(num_runs, name):
        selected_run_ids = np.random.choice(list(ids), size=n, replace=False)
        selected_runs = runs[runs.run_id.isin(selected_run_ids)]
        selected_array_events = array_events[array_events.run_id.isin(selected_run_ids)]
        selected_telescope_events = telescope_events[telescope_events.run_id.isin(selected_run_ids)]

        path = output_basename + '_' + part_name + '.hdf5'
        log.info('Writing {} runs events to: {}'.format(n, path))
        write_data(selected_runs, path, key='runs', use_h5py=True, mode='w')
        write_data(selected_array_events, path, key='array_events', use_h5py=True, mode='a')
        write_data(selected_telescope_events, path, key='telescope_events', use_h5py=True, mode='a')

        with h5py.File(path, 'r+') as f:
            f.attrs['sample_fraction'] = n / n_total

        log.debug(f'selected runs {set(selected_run_ids)}')
        log.debug(f'Runs minus selected runs {ids - set(selected_run_ids)}')
        ids = ids - set(selected_run_ids)


def split_single_telescope_data_chunked(input_path, output_basename, inkey, key, fraction, name, chunksize):

    with h5py.File(input_path, 'r') as f:
        k = next(iter(f[inkey].keys()))
        n_total = f[inkey][k].shape[0]

    log.info('Found a total of {} single-telescope events in the file'.format(n_total))

    n_chunks = int(ceil(n_total / chunksize))
    ids = np.arange(n_total)
    num_ids = split_indices(ids, n_total, fractions=fraction)

    selected_ids = {}
    for n, part_name in zip(num_ids, name):
        selected_ids[part_name] = np.random.choice(ids, size=n, replace=False)
        ids = list(set(ids) - set(selected_ids[part_name]))

        path = output_basename + '_' + part_name + '.hdf5'
        log.info('Will write {n} single-telescope events to {path}'.format(n=n, path=path))

    for chunk in tqdm(range(n_chunks)):
        first = chunk * chunksize
        last = (chunk + 1) * chunksize

        data = read_h5py(input_path, key=inkey, first=first, last=last)
        mode = 'w' if chunk == 0 else 'a'

        for part_name in name:
            mask = (selected_ids[part_name] >= first) & (selected_ids[part_name] < last)
            chunk_ids = selected_ids[part_name][mask]
            selected_data = data.iloc[chunk_ids - first]

            path = output_basename + '_' + part_name + '.hdf5'
            log.debug('Writing {} telescope-array events to: {}'.format(
                len(selected_data), path
            ))
            write_data(selected_data, path, key=key, use_h5py=True, mode=mode)

    for n, part_name in zip(num_ids, name):
        path = output_basename + '_' + part_name + '.hdf5'
        with h5py.File(path, mode='r+') as f:
            f.attrs['sample_fraction'] = n / n_total


def split_single_telescope_data(input_path, output_basename, fmt, inkey, key, fraction, name):

    if fmt in ['hdf5', 'hdf', 'h5']:
        data = read_data(input_path, key=inkey)
    elif fmt == 'csv':
        data = read_data(input_path)

    ids = data.index.values
    n_total = len(data)

    log.info('Found a total of {} single-telescope events in the file'.format(len(data)))

    num_ids = split_indices(ids, n_total, fractions=fraction)

    for n, part_name in zip(num_ids, name):
        selected_ids = np.random.choice(ids, size=n, replace=False)
        selected_data = data.loc[selected_ids]

        if fmt in ['hdf5', 'hdf', 'h5']:
            path = output_basename + '_' + part_name + '.hdf5'
            log.info('Writing {} telescope-array events to: {}'.format(n, path))
            write_data(selected_data, path, key=key, use_h5py=True, mode='w')

            with h5py.File(path, mode='r+') as f:
                f.attrs['sample_fraction'] = n / n_total

        elif fmt == 'csv':
            filename = output_basename + '_' + part_name + '.csv'
            log.info('Writing {} telescope-array events to: {}'.format(n, filename))
            selected_data.to_csv(filename, index=False)

        data = data.loc[list(set(data.index.values) - set(selected_data.index.values))]
        ids = data.index.values
