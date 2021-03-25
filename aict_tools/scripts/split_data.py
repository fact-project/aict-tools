import os

import click
import numpy as np
import logging

import warnings
from math import ceil
from tqdm import tqdm
import h5py

from ..io import (
    read_data,
    copy_group,
    write_hdf,
    set_sample_fraction,
)
from ..logging import setup_logging


log = logging.getLogger(__name__)


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
    '--data-format', '-d', type=click.Choice(['simple', 'CTA']), default='simple',
    show_default=True, help='Which telescope created the data',
)
@click.option(
    '--chunksize',
    type=int,
    help='How many events to read at once, only supported for h5py and single telescopes'
)
@click.option('-s', '--seed', help='Random Seed', type=int, default=0, show_default=True)
@click.option('-v', '--verbose', is_flag=True, help='Verbose log output',)
def main(input_path, output_basename, fraction, name, inkey, key, data_format, chunksize, seed, verbose):
    '''
    Split dataset in INPUT_PATH into multiple parts for given fractions and names
    Outputs hdf5 or csv files to OUTPUT_BASENAME_NAME.FORMAT

    Example call: aict_split_data input.hdf5 output_base -n test -f 0.5 -n train -f 0.5
    '''

    setup_logging(verbose=verbose)
    log.debug("input_path: {}".format(input_path))

    assert len(fraction) == len(name), 'You must give a name for each fraction'

    if sum(fraction) != 1:
        warnings.warn('Fractions do not sum up to 1')

    np.random.seed(seed)

    if data_format == 'simple':
        if chunksize is None:
            split_single_telescope_data(
                input_path, output_basename, inkey, key, fraction, name
            )
        else:
            split_single_telescope_data_chunked(
                input_path, output_basename, inkey, key, fraction, name, chunksize,
            )
    elif data_format == 'CTA':
        split_cta_dl1_data(input_path, output_basename, fraction, name, chunksize)


def split_cta_dl1_data(input_path, output_basename, fraction, name, chunksize=None):
    import tables
    with tables.open_file(input_path) as f:
        obs_ids = f.root.dl1.event.subarray.trigger.col('obs_id')
        event_ids = f.root.dl1.event.subarray.trigger.col('event_id')
        n_total = len(f.root.dl1.event.subarray.trigger)

    log.info('Found a total of {} array events in the file'.format(n_total))
    ids = np.arange(n_total)
    num_ids = split_indices(ids, n_total, fractions=fraction)

    # ctapipe default filters
    filters = tables.Filters(
        complevel=5,  # compression medium, tradeoff between speed and compression
        complib="blosc:zstd",  # use modern zstd algorithm
        fletcher32=True,  # add checksums to data chunks
    )

    for n, part_name in zip(num_ids, name):
        selected_ids = np.random.choice(ids, size=n, replace=False)
        ids = list(set(ids) - set(selected_ids))
        path = output_basename + '_' + part_name + '.dl1.h5'
        log.info('Will write {n} array events to {path}'.format(n=n, path=path))
        obs_ids_part = obs_ids[selected_ids]
        event_ids_part = event_ids[selected_ids]
        with tables.open_file(input_path) as in_, tables.open_file(path, 'a', filters=filters) as out_:
            for member in in_.walk_nodes():
                if isinstance(member, tables.Table):
                    new_table = out_.create_table(
                        member._v_parent._v_pathname,
                        member.name,
                        member.description,
                        createparents=True,
                        expectedrows=min(len(event_ids_part), len(member))
                    )
                    for row in member.iterrows():
                        if 'obs_id' in member.colnames:
                            if row['obs_id'] not in obs_ids_part:
                                continue
                        if 'event_id' in member.colnames:
                            if row['event_id'] not in event_ids_part:
                                continue
                        new_table.append([row[:]])
        set_sample_fraction(path, n / n_total)


def split_single_telescope_data_chunked(
        input_path,
        output_basename,
        inkey,
        key,
        fraction,
        name,
        chunksize
):

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
        log.info(
            'Will write {n} single-telescope events to {path}'.format(n=n, path=path)
        )

    for chunk in tqdm(range(n_chunks)):
        first = chunk * chunksize
        last = (chunk + 1) * chunksize

        data = read_data(input_path, key=inkey, first=first, last=last)
        mode = 'w' if chunk == 0 else 'a'

        for part_name in name:
            mask = (selected_ids[part_name] >= first) & (selected_ids[part_name] < last)
            chunk_ids = selected_ids[part_name][mask]
            selected_data = data.iloc[chunk_ids - first]

            path = output_basename + '_' + part_name + '.hdf5'
            log.debug('Writing {} telescope-array events to: {}'.format(
                len(selected_data), path
            ))
            write_hdf(selected_data, path, table_name=key, mode=mode)

    for n, part_name in zip(num_ids, name):
        path = output_basename + '_' + part_name + '.hdf5'
        set_sample_fraction(path, n / n_total)
        copy_group(input_path, path, 'runs')


def split_single_telescope_data(input_path, output_basename, inkey, key, fraction, name):
    _, file_extension = os.path.splitext(input_path)

    data = read_data(input_path, key=inkey)
    assert len(fraction) == len(name), 'You must give a name for each fraction'

    if sum(fraction) != 1:
        warnings.warn('Fractions do not sum up to 1')

    ids = data.index.values
    n_total = len(data)

    log.info('Found a total of {} single-telescope events in the file'.format(len(data)))

    num_ids = split_indices(ids, n_total, fractions=fraction)

    for n, part_name in zip(num_ids, name):
        selected_ids = np.random.choice(ids, size=n, replace=False)
        selected_data = data.loc[selected_ids]

        path = output_basename + '_' + part_name + file_extension
        log.info('Writing {} telescope-array events to: {}'.format(n, path))
        write_hdf(selected_data, path, table_name=key, mode='w')

        set_sample_fraction(path, n / n_total)
        copy_group(input_path, path, 'runs')

        data = data.loc[list(set(data.index.values) - set(selected_data.index.values))]
        ids = data.index.values
