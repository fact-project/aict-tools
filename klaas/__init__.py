from os import path
import pandas as pd
import json
from sklearn.externals import joblib
from sklearn2pmml import sklearn2pmml, PMMLPipeline
import numpy as np


def write_data(df, file_path, hdf_key='table'):
    name, extension = path.splitext(file_path)
    if extension in ['.hdf', '.hdf5', '.h5']:
        df.to_hdf(file_path, key=hdf_key)
    elif extension == '.json':
        df.to_json(file_path)
    elif extension == '.csv':
        df.to_csv(file_path, delimiter=',', index=False)
    else:
        raise IOError(
                'cannot write tabular data with format {}. Allowed formats: {}'
                .format(extension, 'hdf5, json, csv')
            )


def read_data(file_path, query=None, sample=-1, hdf_key='table'):
    name, extension = path.splitext(file_path)
    if extension in ['.hdf', '.hdf5', '.h5']:
        try:
            df = pd.read_hdf(file_path, key=hdf_key)
        except KeyError:
            df = pd.read_hdf(file_path)
    elif extension == '.json':
        with open(file_path, 'r') as j:
            d = json.load(j)
            df = pd.DataFrame(d)

    elif extension == '.csv':
        df = pd.read_csv(file_path)

    else:
        raise Exception('output file forma: {} not supported'.format(extension))

    if sample > 0:
        print('Taking {} random samples'.format(sample))
        df = df.sample(sample)

    if query:
        print('Quering with string: {}'.format(query))
        df = df.copy().query(query)

    return df


def check_extension(
                        file_path,
                        allowed_extensions=['.hdf', '.hdf5', '.h5', '.json', '.csv'],
                    ):
    p, extension = path.splitext(file_path)
    if extension not in allowed_extensions:
        raise IOError('Allowed formats: {}'.format(allowed_extensions))


def pickle_model(classifier, feature_names, model_path, label_text='label'):
    p, extension = path.splitext(model_path)
    classifier.feature_names = feature_names
    if (extension == '.pmml'):
        print("Pickling model to {} ...".format(model_path))

        joblib.dump(classifier, p + '.pkl', compress=4)

        pipeline = PMMLPipeline([
            ("classifier", classifier)
        ])

        pipeline.target_field = label_text
        pipeline.active_fields = np.array(feature_names)
        sklearn2pmml(pipeline,  model_path)

    else:
        joblib.dump(classifier, model_path, compress=4)
