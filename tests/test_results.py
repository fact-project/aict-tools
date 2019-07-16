import tempfile
import os
from click.testing import CliRunner
import shutil
from traceback import print_exception
import pandas as pd
import numpy as np


def test_energy_regression_results():
    from aict_tools.scripts.train_energy_regressor import main as train
    from aict_tools.io import read_telescope_data
    from aict_tools.apply import predict_energy
    from sklearn.externals import joblib
    from aict_tools.configuration import AICTConfig


    configuration_path = 'examples/full_config.yaml'

    with tempfile.TemporaryDirectory(prefix='aict_tools_test_') as d:
        
        data_path = os.path.join(d, 'gamma.hdf5')
        model_path = os.path.join(d, 'test.pkl')

        shutil.copy('examples/gamma.hdf5', data_path)

        runner = CliRunner()

        result = runner.invoke(
            train,
            [
                configuration_path,
                data_path,
                os.path.join(d, 'test.hdf5'),
                model_path,
            ]
        )

        assert result.exit_code == 0
        
        config = AICTConfig.from_yaml(configuration_path)
        model_config = config.energy

        model = joblib.load(model_path)
        
        df = read_telescope_data(
            data_path, config, model_config.columns_to_read_apply,
            feature_generation_config=model_config.feature_generation
        )
    
        energy_prediction = predict_energy(
            df[model_config.features],
            model,
            log_target=model_config.log_target,
        )
        expectation = pd.read_csv('tests/expected_results.csv')
        np.testing.assert_array_almost_equal(energy_prediction, expectation['energy_prediction'])


def test_seperation_results():
    from aict_tools.scripts.train_separation_model import main as train
    from aict_tools.scripts.apply_energy_regressor import main
    from aict_tools.io import read_telescope_data
    from aict_tools.apply import predict_separator
    from sklearn.externals import joblib
    from aict_tools.configuration import AICTConfig


    configuration_path = 'examples/full_config.yaml'
    expectation = pd.read_csv('tests/expected_results.csv')

    with tempfile.TemporaryDirectory(prefix='aict_tools_test_') as d:
        
        gamma_path = os.path.join(d, 'gamma.hdf5')
        proton_path = os.path.join(d, 'proton.hdf5')
        model_path = os.path.join(d, 'test.pkl')

        shutil.copy('examples/gamma.hdf5', gamma_path)
        shutil.copy('examples/proton.hdf5', proton_path)

        runner = CliRunner()

        result = runner.invoke(
            train,
            [
                configuration_path,
                gamma_path,
                proton_path,
                os.path.join(d, 'test.hdf5'),
                model_path,
            ]
        )

        assert result.exit_code == 0
        
        config = AICTConfig.from_yaml(configuration_path)
        model_config = config.energy
        model = joblib.load(model_path)
        
        df = read_telescope_data(
            proton_path, config, model_config.columns_to_read_apply,
            feature_generation_config=model_config.feature_generation
        )
        protons_prediction = predict_separator(
            df[model_config.features],
            model,
        )


        df = read_telescope_data(
            gamma_path, config, model_config.columns_to_read_apply,
            feature_generation_config=model_config.feature_generation
        )
        gammas_prediction = predict_separator(
            df[model_config.features],
            model,
        )

        np.testing.assert_array_almost_equal(protons_prediction, expectation['separator_prediction_on_protons'])
        np.testing.assert_array_almost_equal(gammas_prediction, expectation['separator_prediction_on_gammas'])
