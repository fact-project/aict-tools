import click
import numpy as np
import logging

from fact.io import read_data, write_data
import warnings


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
    help='HDF5 key for pandas or h5py hdf5 of the input file',
    default='events'
)
@click.option(
    '--key',
    '-k',
    help='Name for the hdf5 group in the output',
    default='events'
)
@click.option(
    '--event_id_key',
    '-d',
    help='Name of the colum containing a unique id for each array-wide event.'
         'If provided it will not split up rows that belong to the same event but to '
         'different telescopes',
    default=None
)
@click.option(
    '--fmt', type=click.Choice(['csv', 'hdf5', 'hdf', 'h5']), default='hdf5',
    help='The output format',
)
@click.option(
    '--use-pandas', is_flag=True, help='Write pandas hdf5 output files',
)
@click.option('-v', '--verbose', help='Verbose log output', type=bool)
def main(input_path, output_basename, fraction, name, inkey, key, event_id_key, fmt, use_pandas, verbose):
    '''
    Split dataset in INPUT_PATH into multiple parts for given fractions and names
    Outputs pandas hdf5 or csv files to OUTPUT_BASENAME_NAME.FORMAT

    Example call: klaas_split_data input.hdf5 output_base -n test -f 0.5 -n train -f 0.5
    '''
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    log = logging.getLogger()
    log.debug("input_path: {}".format(input_path))

    if fmt in ['hdf5', 'hdf', 'h5']:
        data = read_data(input_path, key=inkey)
    elif fmt == 'csv':
        data = read_data(input_path)

    assert len(fraction) == len(name), 'You must give a name for each fraction'

    if sum(fraction) != 1:
        warnings.warn('Fractions do not sum up to 1')

    # set the ids we can split by to be either telescope events or array-wide events
    ids = data.index.values
    n_total = len(data)
    if event_id_key:
        ids = data[event_id_key].unique()
        n_total = len(ids)
        log.info('Found {} telescope-array events'.format(n_total))

    log.info('Found a total of {} single-telescope events in the file'.format(len(data)))

    # find number of events in each split. The last split may contain less events.
    num_ids = [int(round(n_total * f)) for f in fraction]
    if sum(num_ids) > n_total:
        num_ids[-1] -= sum(num_ids) - n_total

    for n, part_name in zip(num_ids, name):
        selected_ids = np.random.choice(ids, size=n, replace=False)
        if event_id_key:
            selected_data = data.loc[data[event_id_key].isin(selected_ids)]
        else:
            selected_data = data.loc[selected_ids]

        if fmt in ['hdf5', 'hdf', 'h5']:
            path = output_basename + '_' + part_name + '.hdf5'
            log.info('Writing {} telescope-array events to: {}'.format(n, path))
            write_data(selected_data, path, key=key, use_hp5y=not use_pandas)

        elif fmt == 'csv':
            filename = output_basename + '_' + part_name + '.csv'
            log.info('Writing {} telescope-array events to: {}'.format(n, filename))
            selected_data.to_csv(filename, index=False)

        data = data.iloc[list(set(data.index.values) - set(selected_data.index))]

        if event_id_key:
            ids = data[event_id_key].unique()
        else:
            ids = data.index.values
