try:
    import onnxruntime
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


class ONNXModel:
    '''wrapper for onnxruntime to use it like an sklearn model'''
    def __init__(self, path):

        self.session = onnxruntime.InferenceSession(path)
        self.input_name = self.session.get_inputs()[0].name
        self.meta = self.session.get_modelmeta().custom_metadata_map
        self.feature_names = self.meta['feature_names'].split(',')
        self._estimator_type = self.meta['model_type']

    def predict(self, X):
        result = self.session.run(None, {self.input_name: X})[0]
        if self._estimator_type == 'classifier':
            return (result[:, 1] >= 0.5).astype('int8')

        return result[:, 0]

    def predict_proba(self, X):
        if not self._estimator_type == 'classifier':
            raise ValueError('Only classifiers can do `predict_proba`')

        probas = self.session.run(None, {self.input_name: X})[0]
        return probas
