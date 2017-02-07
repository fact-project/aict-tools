import click
import sklearn.externals import joblib
import yaml
import logging
import h5py

from ..io import read_data
from ..features import find_used_source_features
from ..apply import predict, predict_off_positions


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('data_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('model_path', type=click.Path(exists=True, dir_okay=False))
@click.option('-k', '--key', help='HDF5 key for pandas or h5py hdf5', default='events')
@click.option(
    '-N', '--chunksize', type=int,
    help='If given, only process the given number of events at once'
)
def main(configuration_path, data_path, model_path, key, chunksize):
    '''
    Apply loaded model to data.

    CONFIGURATION_PATH: Path to the config yaml file.
    DATA_PATH: path to the FACT data.
    MODEL_PATH: Path to the pickled model.

    The program adds the following columns to the inputfile:
        signal_prediction: the output of model.predict_proba for the signal class
    '''
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger()

    with open(configuration_path) as f:
        config = yaml.load(f)

    training_variables = config['training_variables']

    log.info('Loading model')
    model = joblib.load(model_path)
    log.info('Done')

    if chunksize is None:
        log.info('Loading data')
        df_data = read_data(data_path, key=key)
        log.info('Done')

        log.info('Predicting on data...')
        signal_prediction = predict(df_data, model, training_variables)

        with h5py.File(data_path) as f:
            if 'signal_prediction' in f[key].keys():
                log.warning('Overwriting existing signal_prediction')
                f[key]['signal_prediction'] = signal_prediction
            else:
                f[key].create_dataset(
                    'signal_prediction', data=signal_prediction, maxshape=(None, )
                )

        used_source_feautures = find_used_source_features(training_variables)
        if len(used_source_feautures) > 0:
            log.info(
                'Source dependent features used in model, '
                'redoing classification for off regions'
            )

            background_predictions = predict_off_positions(
                df_data, model, training_variables, used_source_feautures
            )

            with h5py.File(data_path) as f:
                for region in range(1, 5):
                    name = 'background_prediction_{}'.format(region)
                    if name in f[key].keys():
                        log.warning('Overwriting existing {}'.format(name))
                        f[key][name] = background_predictions[name]
                    else:
                        f[key].create_dataset(
                            name, data=background_predictions[name], maxshape=(None, )
                        )


if __name__ == '__main__':
    main()
