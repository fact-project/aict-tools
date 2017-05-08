import click
from sklearn.externals import joblib
import yaml
import logging
import h5py
from tqdm import tqdm

from fact.io import read_h5py_chunked
from ..features import find_used_source_features
from ..apply import predict, predict_off_positions
from ..feature_generation import feature_generation


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
        signal_prediction: the output of model.predict_proba for the signal class
    '''
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    log = logging.getLogger()

    with open(configuration_path) as f:
        config = yaml.load(f)

    training_variables = config['training_variables']

    with h5py.File(data_path, 'r+') as f:
        if 'signal_prediction' in f[key].keys():
            if not yes:
                click.confirm(
                    'Dataset "signal_prediction" exists in file, overwrite?',
                    abort=True,
                )
            del f[key]['signal_prediction']

        for region in range(1, 6):
            dataset = 'background_prediction_{}'.format(region)
            if dataset in f[key].keys():
                del f[key][dataset]

    log.info('Loading model')
    model = joblib.load(model_path)
    log.info('Done')

    generation_config = config.get('feature_generation')
    used_source_features = find_used_source_features(
        training_variables
    )
    if generation_config:
        used_source_features = used_source_features.union(
            find_used_source_features(generation_config['needed_keys'])
        )

    if len(used_source_features) > 0:
        log.info(
            'Source dependent features used in model, '
            'redoing classification for off regions'
        )

    needed_features = [
        var + '_Off_{}'.format(region)
        for region in range(1, 6)
        for var in used_source_features
    ]

    if generation_config:
        needed_features.extend(generation_config['needed_keys'])

    df_generator = read_h5py_chunked(
        data_path,
        key=key,
        columns=training_variables + needed_features,
        chunksize=chunksize,
    )

    if generation_config:
        training_variables.extend(generation_config['features'])

    log.info('Predicting on data...')
    for df_data, start, end in tqdm(df_generator):

        if generation_config:
            feature_generation(
                df_data,
                generation_config,
                inplace=True,
            )

        signal_prediction = predict(df_data, model, training_variables)

        with h5py.File(data_path, 'r+') as f:
            if 'signal_prediction' in f[key].keys():
                n_existing = f[key]['signal_prediction'].shape[0]
                n_new = signal_prediction.shape[0]
                f[key]['signal_prediction'].resize(n_existing + n_new, axis=0)
                f[key]['signal_prediction'][start:end] = signal_prediction
            else:
                f[key].create_dataset(
                    'signal_prediction', data=signal_prediction, maxshape=(None, )
                )

        if len(used_source_features) > 0:
            background_predictions = predict_off_positions(
                df_data,
                model,
                training_variables,
                used_source_features,
                generation_config,
            )

            with h5py.File(data_path) as f:
                for region in range(1, 6):
                    name = 'background_prediction_{}'.format(region)
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
