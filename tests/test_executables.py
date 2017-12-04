import tempfile
import os
from click.testing import CliRunner
import shutil


def test_train_regressor():
    from klaas.scripts.train_energy_regressor import main

    with tempfile.TemporaryDirectory(prefix='klaas_test_') as d:
        runner = CliRunner()

        result = runner.invoke(
            main,
            [
                'examples/config_energy.yaml',
                'examples/signal.hdf',
                os.path.join(d, 'test.hdf5'),
                os.path.join(d, 'test.pkl'),
            ]
        )

        if result.exit_code != 0:
            print(result.output)
        assert result.exit_code == 0


def test_apply_regression():
    from klaas.scripts.train_energy_regressor import main as train
    from klaas.scripts.apply_regression_model import main

    with tempfile.TemporaryDirectory(prefix='klaas_test_') as d:
        runner = CliRunner()

        shutil.copy('examples/signal.hdf', os.path.join(d, 'signal.hdf'))

        result = runner.invoke(
            train,
            [
                'examples/config_energy.yaml',
                os.path.join(d, 'signal.hdf'),
                os.path.join(d, 'test.hdf5'),
                os.path.join(d, 'test.pkl'),
            ]
        )
        if result.exit_code != 0:
            print(result.output)
        assert result.exit_code == 0

        result = runner.invoke(
            main,
            [
                'examples/config_energy.yaml',
                os.path.join(d, 'signal.hdf'),
                os.path.join(d, 'test.pkl'),
                '--yes',
            ]
        )

        if result.exit_code != 0:
            print(result.output)

        assert result.exit_code == 0


def test_train_separator():
    from klaas.scripts.train_separation_model import main

    with tempfile.TemporaryDirectory(prefix='klaas_test_') as d:
        runner = CliRunner()

        result = runner.invoke(
            main,
            [
                'examples/config_separator.yaml',
                'examples/signal.hdf',
                'examples/background.hdf',
                os.path.join(d, 'test.hdf5'),
                os.path.join(d, 'test.pkl'),
            ]
        )

        if result.exit_code != 0:
            print(result.output)
        assert result.exit_code == 0


def test_apply_separator():
    from klaas.scripts.train_separation_model import main as train
    from klaas.scripts.apply_separation_model import main as apply_model

    with tempfile.TemporaryDirectory(prefix='klaas_test_') as d:

        shutil.copy('examples/signal.hdf', os.path.join(d, 'signal.hdf'))

        runner = CliRunner()
        result = runner.invoke(
            train,
            [
                'examples/config_separator.yaml',
                'examples/signal.hdf',
                'examples/background.hdf',
                os.path.join(d, 'test.hdf5'),
                os.path.join(d, 'test.pkl'),
            ]
        )
        if result.exit_code != 0:
            print(result.output)
        assert result.exit_code == 0

        result = runner.invoke(
            apply_model,
            [
                'examples/config_separator.yaml',
                os.path.join(d, 'signal.hdf'),
                os.path.join(d, 'test.pkl'),
                '--yes',
            ]
        )

        if result.exit_code != 0:
            print(result.output)
        assert result.exit_code == 0


def test_train_disp():
    from klaas.scripts.train_disp_regressor import main as train

    with tempfile.TemporaryDirectory(prefix='klaas_test_') as d:

        shutil.copy('examples/signal.hdf', os.path.join(d, 'signal.hdf'))

        runner = CliRunner()
        result = runner.invoke(
            train,
            [
                'examples/config_source.yaml',
                'examples/signal.hdf',
                os.path.join(d, 'test.hdf5'),
                os.path.join(d, 'disp.pkl'),
                os.path.join(d, 'sign.pkl'),
            ]
        )
        if result.exit_code != 0:
            print(result.output)
        assert result.exit_code == 0


def test_apply_disp():
    from klaas.scripts.train_disp_regressor import main as train
    from klaas.scripts.apply_disp_regressor import main as apply_model

    with tempfile.TemporaryDirectory(prefix='klaas_test_') as d:

        shutil.copy('examples/signal.hdf', os.path.join(d, 'signal.hdf'))

        runner = CliRunner()
        result = runner.invoke(
            train,
            [
                'examples/config_source.yaml',
                'examples/signal.hdf',
                os.path.join(d, 'test.hdf5'),
                os.path.join(d, 'disp.pkl'),
                os.path.join(d, 'sign.pkl'),
            ]
        )
        if result.exit_code != 0:
            print(result.output)
        assert result.exit_code == 0

        result = runner.invoke(
            apply_model,
            [
                'examples/config_source.yaml',
                os.path.join(d, 'signal.hdf'),
                os.path.join(d, 'disp.pkl'),
                os.path.join(d, 'sign.pkl'),
                '--yes',
            ]
        )

        if result.exit_code != 0:
            print(result.output)
        assert result.exit_code == 0
