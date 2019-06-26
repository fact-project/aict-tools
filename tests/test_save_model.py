from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import tempfile
from pytest import importorskip
import numpy as np
import os
from sklearn.externals import joblib
import pandas as pd


y_clf = np.random.randint(0, 2, 100)
y_reg = np.random.uniform(0, 1, 100)
X_clf = np.random.normal(y_clf, 0.5, size=(5, 100)).T.astype('float32')
X_reg = np.random.normal(y_reg, 0.5, size=(5, 100)).T.astype('float32')
feature_names = list('abcde')

clf = RandomForestClassifier(n_estimators=10)
clf.fit(X_clf, y_clf)
reg = RandomForestRegressor(n_estimators=10)
reg.fit(X_reg, y_reg)


def test_pickle():
    from aict_tools.io import save_model

    with tempfile.TemporaryDirectory(prefix='aict_tools_test_') as tmpdir:
        model_path = os.path.join(tmpdir, 'model.pkl')
        save_model(clf, feature_names, model_path, label_text='classifier')

        clf_load = joblib.load(model_path)
        assert clf_load.feature_names == feature_names

    with tempfile.TemporaryDirectory(prefix='aict_tools_test_') as tmpdir:
        model_path = os.path.join(tmpdir, 'model.pkl')
        save_model(reg, feature_names, model_path, label_text='regressor')

        reg_load = joblib.load(model_path)
        assert reg_load.feature_names == feature_names
        assert np.all(reg.predict(X_reg) == reg_load.predict(X_reg))
        assert np.all(clf.predict(X_clf) == clf_load.predict(X_clf))


def test_pmml():
    importorskip('sklearn2pmml')
    jpmml_evaluator = importorskip('jpmml_evaluator')
    from jpmml_evaluator.pyjnius import jnius_configure_classpath, PyJNIusBackend
    from aict_tools.io import save_model

    jnius_configure_classpath()
    backend = PyJNIusBackend()

    with tempfile.TemporaryDirectory(prefix='aict_tools_test_') as tmpdir:
        model_path = os.path.join(tmpdir, 'model.pmml')
        save_model(clf, feature_names, model_path, label_text='classifier')

        evaluator = jpmml_evaluator.make_evaluator(backend, model_path).verify()
        assert [i.getName() for i in evaluator.getInputFields()] == feature_names

        df = evaluator.evaluateAll(pd.DataFrame(dict(zip(feature_names, X_clf.T))))
        assert np.all(np.isclose(df['probability(1)'], clf.predict_proba(X_clf)[:, 1]))

        # make sure pickle is also saved
        clf_load = joblib.load(model_path.replace('.pmml', '.pkl'))
        assert clf_load.feature_names == feature_names
        assert np.all(clf.predict(X_clf) == clf_load.predict(X_clf))

    with tempfile.TemporaryDirectory(prefix='aict_tools_test_') as tmpdir:
        model_path = os.path.join(tmpdir, 'model.pmml')
        save_model(reg, feature_names, model_path, label_text='regressor')

        evaluator = jpmml_evaluator.make_evaluator(backend, model_path).verify()
        assert [i.getName() for i in evaluator.getInputFields()] == feature_names

        df = evaluator.evaluateAll(pd.DataFrame(dict(zip(feature_names, X_reg.T))))
        assert np.all(np.isclose(df['regressor'], reg.predict(X_reg)))

        # make sure pickle is also saved
        reg_load = joblib.load(model_path.replace('.pmml', '.pkl'))
        assert reg_load.feature_names == feature_names
        assert np.all(reg.predict(X_reg) == reg_load.predict(X_reg))


def test_onnx():
    importorskip('skl2onnx')
    onnxruntime = importorskip('onnxruntime')
    from aict_tools.io import save_model

    with tempfile.TemporaryDirectory(prefix='aict_tools_test_') as tmpdir:
        model_path = os.path.join(tmpdir, 'model.onnx')
        save_model(clf, feature_names, model_path, label_text='classifier')

        session = onnxruntime.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name

        probas_onnx = session.run(None, {input_name: X_clf})[0]
        probas_skl = clf.predict_proba(X_clf)

        assert np.all(np.isclose(probas_onnx, probas_skl, rtol=1e-2, atol=1e-4))

        # make sure pickle is also saved
        clf_load = joblib.load(model_path.replace('.onnx', '.pkl'))
        assert clf_load.feature_names == feature_names
        assert np.all(clf.predict(X_clf) == clf_load.predict(X_clf))

    with tempfile.TemporaryDirectory(prefix='aict_tools_test_') as tmpdir:
        model_path = os.path.join(tmpdir, 'model.onnx')
        save_model(reg, feature_names, model_path, label_text='regressor')

        session = onnxruntime.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name

        pred_onnx = session.run(None, {input_name: X_reg})[0][:, 0]
        pred_skl = reg.predict(X_reg)

        assert np.all(np.isclose(pred_onnx, pred_skl))

        # make sure pickle is also saved
        reg_load = joblib.load(model_path.replace('.onnx', '.pkl'))
        assert reg_load.feature_names == feature_names
        assert np.all(reg.predict(X_reg) == reg_load.predict(X_reg))
