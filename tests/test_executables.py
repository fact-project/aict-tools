import tempfile
import os
from click.testing import CliRunner
import shutil
from traceback import print_exception
import h5py


def test_train_regressor():
    from aict_tools.scripts.train_energy_regressor import main

    with tempfile.TemporaryDirectory(prefix='aict_tools_test_') as d:
        runner = CliRunner()

        result = runner.invoke(
            main,
            [
                'examples/config_energy.yaml',
                'examples/gamma.hdf5',
                os.path.join(d, 'test.hdf5'),
                os.path.join(d, 'test.pkl'),
            ]
        )

        if result.exit_code != 0:
            print(result.output)
            print_exception(*result.exc_info)
        assert result.exit_code == 0


def test_apply_regression():
    from aict_tools.scripts.train_energy_regressor import main as train
    from aict_tools.scripts.apply_energy_regressor import main

    with tempfile.TemporaryDirectory(prefix='aict_tools_test_') as d:
        runner = CliRunner()

        shutil.copy('examples/gamma.hdf5', os.path.join(d, 'gamma.hdf5'))

        result = runner.invoke(
            train,
            [
                'examples/config_energy.yaml',
                os.path.join(d, 'gamma.hdf5'),
                os.path.join(d, 'test.hdf5'),
                os.path.join(d, 'test.pkl'),
            ]
        )
        if result.exit_code != 0:
            print(result.output)
            print_exception(*result.exc_info)
        assert result.exit_code == 0

        result = runner.invoke(
            main,
            [
                'examples/config_energy.yaml',
                os.path.join(d, 'gamma.hdf5'),
                os.path.join(d, 'test.pkl'),
                '--yes',
            ]
        )

        if result.exit_code != 0:
            print(result.output)
            print_exception(*result.exc_info)

        assert result.exit_code == 0


def test_train_separator():
    from aict_tools.scripts.train_separation_model import main

    with tempfile.TemporaryDirectory(prefix='aict_tools_test_') as d:
        runner = CliRunner()

        result = runner.invoke(
            main,
            [
                'examples/config_separator.yaml',
                'examples/gamma.hdf5',
                'examples/proton.hdf5',
                os.path.join(d, 'test.hdf5'),
                os.path.join(d, 'test.pkl'),
            ]
        )

        if result.exit_code != 0:
            print(result.output)
            print_exception(*result.exc_info)
        assert result.exit_code == 0


def test_apply_separator():
    from aict_tools.scripts.train_separation_model import main as train
    from aict_tools.scripts.apply_separation_model import main as apply_model

    with tempfile.TemporaryDirectory(prefix='aict_tools_test_') as d:
        shutil.copy('examples/gamma.hdf5', os.path.join(d, 'gamma.hdf5'))

        runner = CliRunner()
        result = runner.invoke(
            train,
            [
                'examples/config_separator.yaml',
                'examples/gamma.hdf5',
                'examples/proton.hdf5',
                os.path.join(d, 'test.hdf5'),
                os.path.join(d, 'test.pkl'),
            ]
        )
        if result.exit_code != 0:
            print(result.output)
            print_exception(*result.exc_info)
        assert result.exit_code == 0

        result = runner.invoke(
            apply_model,
            [
                'examples/config_separator.yaml',
                os.path.join(d, 'gamma.hdf5'),
                os.path.join(d, 'test.pkl'),
                '--yes',
            ]
        )

        if result.exit_code != 0:
            print(result.output)
            print_exception(*result.exc_info)
        assert result.exit_code == 0


def test_train_disp():
    from aict_tools.scripts.train_disp_regressor import main as train

    with tempfile.TemporaryDirectory(prefix='aict_tools_test_') as d:

        runner = CliRunner()
        result = runner.invoke(
            train,
            [
                'examples/config_source.yaml',
                'examples/gamma_diffuse.hdf5',
                os.path.join(d, 'test.hdf5'),
                os.path.join(d, 'disp.pkl'),
                os.path.join(d, 'sign.pkl'),
            ]
        )
        if result.exit_code != 0:
            print(result.output)
            print_exception(*result.exc_info)
        assert result.exit_code == 0


def test_apply_disp():
    from aict_tools.scripts.train_disp_regressor import main as train
    from aict_tools.scripts.apply_disp_regressor import main as apply_model

    with tempfile.TemporaryDirectory(prefix='aict_tools_test_') as d:

        shutil.copy('examples/gamma.hdf5', os.path.join(d, 'gamma.hdf5'))

        runner = CliRunner()
        result = runner.invoke(
            train,
            [
                'examples/config_source.yaml',
                'examples/gamma_diffuse.hdf5',
                os.path.join(d, 'test.hdf5'),
                os.path.join(d, 'disp.pkl'),
                os.path.join(d, 'sign.pkl'),
            ]
        )
        if result.exit_code != 0:
            print(result.output)
            print_exception(*result.exc_info)
        assert result.exit_code == 0

        result = runner.invoke(
            apply_model,
            [
                'examples/config_source.yaml',
                os.path.join(d, 'gamma.hdf5'),
                os.path.join(d, 'disp.pkl'),
                os.path.join(d, 'sign.pkl'),
                '--yes',
            ]
        )

        if result.exit_code != 0:
            print(result.output)
            print_exception(*result.exc_info)
        assert result.exit_code == 0


def test_to_dl3():
    from aict_tools.scripts.train_disp_regressor import main as train_disp
    from aict_tools.scripts.train_energy_regressor import main as train_energy
    from aict_tools.scripts.train_separation_model import main as train_separator
    from aict_tools.scripts.fact_to_dl3 import main as to_dl3

    with tempfile.TemporaryDirectory(prefix='aict_tools_test_') as d:

        runner = CliRunner()

        result = runner.invoke(
            train_disp,
            [
                'examples/full_config.yaml',
                'examples/gamma_diffuse.hdf5',
                os.path.join(d, 'disp_performance.hdf5'),
                os.path.join(d, 'disp.pkl'),
                os.path.join(d, 'sign.pkl'),
            ]
        )
        if result.exit_code != 0:
            print(result.output)
            print_exception(*result.exc_info)
        assert result.exit_code == 0

        result = runner.invoke(
            train_energy,
            [
                'examples/full_config.yaml',
                'examples/gamma.hdf5',
                os.path.join(d, 'regressor_performance.hdf5'),
                os.path.join(d, 'regressor.pkl'),
            ]
        )

        if result.exit_code != 0:
            print(result.output)
            print_exception(*result.exc_info)
        assert result.exit_code == 0

        result = runner.invoke(
            train_separator,
            [
                'examples/full_config.yaml',
                'examples/gamma.hdf5',
                'examples/proton.hdf5',
                os.path.join(d, 'separator_performance.hdf5'),
                os.path.join(d, 'separator.pkl'),
            ]
        )

        if result.exit_code != 0:
            print(result.output)
            print_exception(*result.exc_info)
        assert result.exit_code == 0

        result = runner.invoke(
            to_dl3,
            [
                'examples/full_config.yaml',
                'examples/crab.hdf5',
                os.path.join(d, 'separator.pkl'),
                os.path.join(d, 'regressor.pkl'),
                os.path.join(d, 'disp.pkl'),
                os.path.join(d, 'sign.pkl'),
                os.path.join(d, 'crab_dl3.hdf5'),
            ]
        )

        if result.exit_code != 0:
            print(result.output)
            print_exception(*result.exc_info)
        assert result.exit_code == 0

        result = runner.invoke(
            to_dl3,
            [
                'examples/full_config.yaml',
                'examples/gamma.hdf5',
                os.path.join(d, 'separator.pkl'),
                os.path.join(d, 'regressor.pkl'),
                os.path.join(d, 'disp.pkl'),
                os.path.join(d, 'sign.pkl'),
                os.path.join(d, 'gamma_dl3.hdf5'),
            ]
        )

        if result.exit_code != 0:
            print(result.output)
            print_exception(*result.exc_info)
        assert result.exit_code == 0


def test_split_data_executable():
    from aict_tools.scripts.split_data import main as split

    with tempfile.TemporaryDirectory(prefix='aict_tools_test_') as d:

        shutil.copy('examples/gamma.hdf5', os.path.join(d, 'gamma.hdf5'))

        runner = CliRunner()
        result = runner.invoke(
            split,
            [
                os.path.join(d, 'gamma.hdf5'),
                os.path.join(d, 'signal'),
                '-ntest',  # no spaces here. maybe a bug in click?
                '-f0.75',
                '-ntrain',
                '-f0.25',
            ]
        )
        if result.exit_code != 0:
            print(result.output)
            print_exception(*result.exc_info)
        assert result.exit_code == 0

        print(os.listdir(d))
        test_path = os.path.join(d, 'signal_test.hdf5')
        assert os.path.isfile(test_path)

        with h5py.File(test_path, 'r') as f:
            assert f.attrs['sample_fraction'] == 0.75

        train_path = os.path.join(d, 'signal_train.hdf5')
        assert os.path.isfile(train_path)

        with h5py.File(train_path, 'r') as f:
            assert f.attrs['sample_fraction'] == 0.25


def test_split_data_executable_chunked():
    from aict_tools.scripts.split_data import main as split

    with tempfile.TemporaryDirectory(prefix='aict_tools_test_') as d:

        shutil.copy('examples/gamma.hdf5', os.path.join(d, 'gamma.hdf5'))

        runner = CliRunner()
        result = runner.invoke(
            split,
            [
                os.path.join(d, 'gamma.hdf5'),
                os.path.join(d, 'signal'),
                '-ntest',  # no spaces here. maybe a bug in click?
                '-f0.75',
                '-ntrain',
                '-f0.25',
                '--chunksize=100',
            ]
        )
        if result.exit_code != 0:
            print(result.output)
            print_exception(*result.exc_info)
        assert result.exit_code == 0

        print(os.listdir(d))
        test_path = os.path.join(d, 'signal_test.hdf5')
        assert os.path.isfile(test_path)

        with h5py.File(test_path, 'r') as f:
            assert f.attrs['sample_fraction'] == 0.75

        train_path = os.path.join(d, 'signal_train.hdf5')
        assert os.path.isfile(train_path)

        with h5py.File(train_path, 'r') as f:
            assert f.attrs['sample_fraction'] == 0.25
