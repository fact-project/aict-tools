from os import path
from sklearn.externals import joblib
from sklearn2pmml import sklearn2pmml, PMMLPipeline
import logging
import numpy as np

__all__ = ['pickle_model']


log = logging.getLogger(__name__)


def pickle_model(classifier, feature_names, model_path, label_text='label'):
    p, extension = path.splitext(model_path)
    classifier.feature_names = feature_names

    if (extension == '.pmml'):
        joblib.dump(classifier, p + '.pkl', compress=4)

        pipeline = PMMLPipeline([
            ('classifier', classifier)
        ])
        pipeline.target_field = label_text
        pipeline.active_fields = np.array(feature_names)
        sklearn2pmml(pipeline, model_path)

    else:
        joblib.dump(classifier, model_path, compress=4)
