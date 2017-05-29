import click
from sklearn.externals import joblib
import yaml
import logging
import h5py
from tqdm import tqdm

from fact.io import read_h5py_chunked
from ..features import find_used_source_features
from ..apply import predict, predict_off_positions


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('data_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('model_path', type=click.Path(exists=True, dir_okay=False))
@click.option('-k', '--key', help='HDF5 key for pandas or h5py hdf5', default='events')
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

    If source dependent features are part of the training variables the
    columns
        <class_name>_prediction_off_(1...n)
    will also be added.

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

        for region in range(1, 6):
            dataset = '{}_off_{}'.format(prediction_column_name, region)
            if dataset in f[key].keys():
                del f[key][dataset]

    log.info('Loading model')
    model = joblib.load(model_path)
    log.info('Done')

    used_source_features = find_used_source_features(training_variables)
    if len(used_source_features) > 0:
        log.info(
            'Source dependent features used in model, '
            'redoing classification for off regions'
        )

    needed_features = [
        var + '_off_{}'.format(region)
        for region in range(1, 6)
        for var in used_source_features
    ]

    df_generator = read_h5py_chunked(
        data_path,
        key=key,
        columns=training_variables + needed_features,
        chunksize=chunksize,
    )

    log.info('Predicting on data...')
    for df_data, start, end in tqdm(df_generator):

        signal_prediction = predict(df_data, model, training_variables)

        with h5py.File(data_path, 'r+') as f:
            if prediction_column_name in f[key].keys():
                n_existing = f[key][prediction_column_name].shape[0]
                n_new = signal_prediction.shape[0]
                f[key][prediction_column_name].resize(n_existing + n_new, axis=0)
                f[key][prediction_column_name][start:end] = signal_prediction
            else:
                f[key].create_dataset(
                    prediction_column_name, data=signal_prediction, maxshape=(None, )
                )

        if len(used_source_features) > 0:
            background_predictions = predict_off_positions(
                df_data,
                model,
                training_variables,
                used_source_features,
            )

            with h5py.File(data_path) as f:
                for region in range(1, 6):
                    name = '{}_off_{}'.format(prediction_column_name, region)
                    if name in f[key].keys():
                        n_existing = f[key][name].shape[0]
                        n_new = background_predictions[name].shape[0]
                        f[key][name].resize(n_existing + n_new, axis=0)
                        f[key][name][start:end] = background_predictions[name]
                    else:
                        f[key].create_dataset(
                            name, data=background_predictions[name], maxshape=(None, )
                        )


if __name__ == '__main__':
    main()
