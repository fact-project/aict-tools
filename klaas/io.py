from os import path
from sklearn_pandas import DataFrameMapper
from sklearn.externals import joblib
from sklearn2pmml import sklearn2pmml
import logging

__all__ = ['pickle_model']


log = logging.getLogger(__name__)


def pickle_model(classifier, feature_names, model_path, label_text='label'):
    p, extension = path.splitext(model_path)
    classifier.feature_names = feature_names
    if (extension == '.pmml'):
        mapper = DataFrameMapper([
            (feature_names, None),
            (label_text, None),
        ])

        joblib.dump(classifier, p + '.pkl', compress=4)
        sklearn2pmml(classifier, mapper,  model_path)

    else:
        joblib.dump(classifier, model_path, compress=4)
