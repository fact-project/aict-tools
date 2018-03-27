import tempfile
import os
from click.testing import CliRunner
import shutil
from traceback import print_exception


def test_train_regressor():
    from klaas.scripts.train_energy_regressor import main

    with tempfile.TemporaryDirectory(prefix='klaas_test_') as d:
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
    from klaas.scripts.train_energy_regressor import main as train
    from klaas.scripts.apply_regression_model import main

    with tempfile.TemporaryDirectory(prefix='klaas_test_') as d:
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
    from klaas.scripts.train_separation_model import main

    with tempfile.TemporaryDirectory(prefix='klaas_test_') as d:
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
    from klaas.scripts.train_separation_model import main as train
    from klaas.scripts.apply_separation_model import main as apply_model

    with tempfile.TemporaryDirectory(prefix='klaas_test_') as d:
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
    from klaas.scripts.train_disp_regressor import main as train

    with tempfile.TemporaryDirectory(prefix='klaas_test_') as d:

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
    from klaas.scripts.train_disp_regressor import main as train
    from klaas.scripts.apply_disp_regressor import main as apply_model

    with tempfile.TemporaryDirectory(prefix='klaas_test_') as d:

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


def test_split_data_executable():
    from klaas.scripts.split_data import main as split

    with tempfile.TemporaryDirectory(prefix='klaas_test_') as d:

        shutil.copy('examples/gamma.hdf5', os.path.join(d, 'gamma.hdf5'))

        runner = CliRunner()
        result = runner.invoke(
            split,
            [
                os.path.join(d, 'gamma.hdf5'),
                os.path.join(d, 'signal'),
                '-ntest',  # no spaces here. maybe a bug in click?
                '-f0.5',
                '-ntrain',
                '-f0.5',
            ]
        )
        if result.exit_code != 0:
            print(result.output)
            print_exception(*result.exc_info)
        assert result.exit_code == 0

        print(os.listdir(d))
        test_path = os.path.join(d, 'signal_test.hdf5')
        assert os.path.isfile(test_path)

        train_path = os.path.join(d, 'signal_train.hdf5')
        assert os.path.isfile(train_path)
