import click
import numpy as np
import h5py

from fact.io import read_data

import logging


@click.command()
@click.argument('inputfile', type=click.Path(exists=False, dir_okay=False))
@click.argument('outputfile', type=click.Path(exists=False, dir_okay=False))
@click.option('-i', '--input-key', help='HDF5 key in pandas inputfile', default='table')
@click.option('-o', '--output-key', help='HDF5 key in h5py outputfile', default='events')
@click.option('-v', '--verbose', is_flag=True)
def main(inputfile, outputfile, input_key, output_key, verbose):
    '''
    Convert a pandas style hdf5 file to a h5py style hdf5 file

    INPUTFILE: A pandas style hdf5 file
    OUTPUTFILE: path for the output file in h5py format
    '''

    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    log = logging.getLogger()

    log.info('Reading input data')
    df = read_data(inputfile, key=input_key)
    log.info('done')

    with h5py.File(outputfile, 'w') as f:
        g = f.create_group(output_key)

        for column, data in df.items():

            if data.dtype == object:
                log.debug('Columns has dtype object: {}'.format(column))

                if isinstance(data.iloc[0], str):
                    log.debug('Columns is str: {}'.format(column))
                    dt = h5py.special_dtype(vlen=str)
                    g.create_dataset(
                        column, data=data.values, dtype=dt, maxshape=(None, )
                    )

                elif isinstance(data.iloc[0], list):
                    log.debug('Columns is list: {}'.format(column))
                    array = np.array([o for o in data.values])
                    shape = list(array.shape)
                    shape[0] = None
                    g.create_dataset(column, data=array, maxshape=tuple(shape))

                else:
                    log.warn('skipping object type column {}'.format(column))
            else:
                log.debug('Writing out {}'.format(column))
                g.create_dataset(column, data=data.values, maxshape=(None, ))
