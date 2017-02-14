import click
import numpy as np

from ..io import read_data
import warnings


@click.command()
@click.argument('input_path', type=click.Path(exists=True, file_okay=True))
@click.argument('output_basename')
@click.option(
    '--fraction', '-f', multiple=True, type=float,
    help='Fraction of events to use for this part'
)
@click.option('--name', '-n', multiple=True, help='name for one dataset')
@click.option('--key', '-k', help='Name for the hdf5 group in the output', default='data')
@click.option(
    '--fmt', type=click.Choice(['csv', 'hdf5']), default='hdf5',
    help='The output format',
)
def main(input_path, output_basename, fraction, name, key, fmt):
    '''
    Split dataset in INPUT_PATH into multiple parts for given fractions and names
    Outputs pandas hdf5 or csv files to OUTPUT_BASENAME_NAME.FORMAT
    '''

    data = read_data(input_path)

    assert len(fraction) == len(name), 'You must give a name for each fraction'

    if sum(fraction) != 1:
        warnings.warn('Fractions do not sum up to 1')

    n_total = len(data)
    print('Found a total of {} events'.format(n_total))

    num_events = [int(round(n_total * f)) for f in fraction]

    for n, part_name in zip(num_events, name):
        print(part_name, ': ', n, sep='')

        all_idx = np.arange(len(data))
        selected = np.random.choice(all_idx, size=n, replace=False)

        print(len(selected))

        if fmt == 'hdf5':
            data.iloc[selected].to_hdf(
                output_basename + '_' + part_name + '.hdf5',
                key=key
            )
        elif fmt == 'csv':
            data.iloc[selected].to_csv(output_basename + '_' + part_name + '.csv')

        data = data.iloc[list(set(all_idx) - set(selected))]
