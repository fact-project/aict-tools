import sys
import tempfile
import os
from click.testing import CliRunner


def test_train_regressor():
    from klaas.scripts.train_energy_regressor import main

    with tempfile.TemporaryDirectory(prefix='klaas_test_') as d:
        runner = CliRunner()

        result = runner.invoke(
            main,
            [
                'examples/config_regressor.yaml',
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

        result = runner.invoke(
            train,
            [
                'examples/config_regressor.yaml',
                'examples/signal.hdf',
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
                'examples/config_regressor.yaml',
                'examples/signal.hdf',
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
    from klaas.scripts.apply_separation_model import main

    with tempfile.TemporaryDirectory(prefix='klaas_test_') as d:
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
            main,
            [
                'examples/config_separator.yaml',
                'examples/signal.hdf',
                os.path.join(d, 'test.pkl'),
                '--yes',
            ]
        )

        if result.exit_code != 0:
            print(result.output)
        assert result.exit_code == 0
