import click
from sklearn.externals import joblib
import yaml
import logging
import h5py
from tqdm import tqdm

from fact.io import read_h5py_chunked

from ..features import find_used_source_features
from ..apply import predict_separator
from ..io import append_to_h5py


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('data_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('model_path', type=click.Path(exists=True, dir_okay=False))
@click.option('-k', '--key', help='HDF5 key for h5py hdf5', default='events')
@click.option('-v', '--verbose', help='Verbose log output', is_flag=True)
@click.option(
    '-N', '--chunksize', type=int,
    help='If given, only process the given number of events at once'
)
@click.option('-y', '--yes', help='Do not prompt for overwrites', is_flag=True)
def main(configuration_path, data_path, model_path, key, chunksize, yes, verbose):
    '''
    Apply loaded model to data.

    CONFIGURATION_PATH: Path to the config yaml file.
    DATA_PATH: path to the FACT data.
    MODEL_PATH: Path to the pickled model.

    The program adds the following columns to the inputfile:
        <class_name>_prediction: the output of model.predict_proba for the
        class name given in the config file.

    If the class name is not given in the config file, the default value of "gamma"
    will be used.
    '''
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    log = logging.getLogger()

    with open(configuration_path) as f:
        config = yaml.load(f)

    training_variables = config['training_variables']

    prediction_column_name = config.get('class_name', 'gamma') + '_prediction'

    with h5py.File(data_path, 'r+') as f:
        if prediction_column_name in f[key].keys():
            if not yes:
                click.confirm(
                    'Column "{}" exists in file, overwrite?'.format(prediction_column_name),
                    abort=True,
                )
            del f[key][prediction_column_name]


    log.info('Loading model')
    model = joblib.load(model_path)
    log.info('Done')

    generation_config = config.get('feature_generation')
    if len(find_used_source_features(training_variables, generation_config)) > 0:
        raise click.ClickException(
            'Using source dependent features in the model is not supported'
        )

    needed_features = training_variables.copy()
    if generation_config:
        needed_features.extend(generation_config['needed_keys'])

    df_generator = read_h5py_chunked(
        data_path,
        key=key,
        columns=needed_features,
        chunksize=chunksize,
        mode='r+'
    )

    log.info('Predicting on data...')
    for df_data, start, end in tqdm(df_generator):

        signal_prediction = predict_separator(df_data, model, config)
        with h5py.File(data_path, 'r+') as f:
            append_to_h5py(f, signal_prediction, key, prediction_column_name)


if __name__ == '__main__':
    main()
