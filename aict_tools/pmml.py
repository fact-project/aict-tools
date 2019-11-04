import pandas as pd
try:
    import jpmml_evaluator
    from jpmml_evaluator.pyjnius import jnius_configure_classpath, PyJNIusBackend
    jnius_configure_classpath()
    HAS_PMML = True
except ImportError:
    HAS_PMML = False


class PMMLModel:
    '''
    Wrapper around jpmml_evaluator so that a pmml model can be used
    to predict just like a sklearn model
    '''
    def __init__(self, path):
        if not HAS_PMML:
            raise ImportError(
                'You need `jpmml_evaluator` to load pmml models.'
                'Use pip install -U aict-tools[pmml] or [all]'
            )
        self.backend = PyJNIusBackend()
        self.evaluator = jpmml_evaluator.make_evaluator(self.backend, path).verify()
        self.feature_names = sorted([
            i.getName() for i in self.evaluator.getInputFields()
        ])

        names = [o.getName() for o in self.evaluator.getTargetFields()]
        if len(names) > 1:
            raise ValueError('Model has more than one output')
        self.target_name = names[0]

    def predict(self, X):
        feature_df = pd.DataFrame(dict(zip(self.feature_names, X.T)))
        result = self.evaluator.evaluateAll(feature_df)
        return result[self.target_name].to_numpy()

    def predict_proba(self, X):
        feature_df = pd.DataFrame(dict(zip(self.feature_names, X.T)))
        result = self.evaluator.evaluateAll(feature_df)
        return result[['probability(0)', 'probability(1)']].to_numpy()
