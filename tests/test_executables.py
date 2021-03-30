import tempfile
import os
from click.testing import CliRunner
import shutil
from traceback import print_exception
import h5py
from pytest import importorskip
import pytest
import pandas as pd
import numpy as np


class DateNotModified:
    def __init__(self, files):
        if isinstance(files, str):
            self.files = [files]
        else:
            self.files = files

    def __enter__(self):
        self.times = {f: os.path.getmtime(f) for f in self.files}
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for f, t in self.times.items():
            assert t == os.path.getmtime(f), 'timestamp of "{}" was modified'.format(f)


def test_apply_cuts():
    from aict_tools.scripts.apply_cuts import main

    with tempfile.TemporaryDirectory(prefix="aict_tools_test_") as d:
        runner = CliRunner()
        output_file = os.path.join(d, "crab_cuts.hdf5")
        input_file = "examples/crab.hdf5"

        with DateNotModified(input_file):
            result = runner.invoke(
                main,
                [
                    "examples/quality_cuts.yaml",
                    input_file,
                    output_file,
                ],
            )

            if result.exit_code != 0:
                print(result.output)
                print_exception(*result.exc_info)

            assert result.exit_code == 0
            with h5py.File(output_file, "r") as f:
                assert "events" in f
                assert "runs" in f


def test_apply_cuts_cta():
    from aict_tools.scripts.apply_cuts import main

    with tempfile.TemporaryDirectory(prefix="aict_tools_test_") as d:
        runner = CliRunner()
        input_file = "examples/cta_gammas_diffuse.dl1.h5"
        output_file = os.path.join(d, "cta_gammas_diffuse_cuts.dl1.h5")

        with DateNotModified(input_file):
            result = runner.invoke(
                main,
                [
                    "examples/cta_full_config.yaml",
                    input_file,
                    output_file,
                ],
            )

            if result.exit_code != 0:
                print(result.output)
                print_exception(*result.exc_info)

            assert result.exit_code == 0

            with h5py.File(output_file, "r") as out, h5py.File(input_file, "r") as in_:
                assert "dl1" in in_
                assert "dl1" in out


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory(prefix="aict_tools_test_") as d:
        for name in ("gamma", "gamma_diffuse", "proton"):
            shutil.copy(f"examples/{name}.hdf5", os.path.join(d, f"{name}.hdf5"))
        for name in ("gammas", "gammas_diffuse", "protons"):
            shutil.copy(
                f"examples/cta_{name}.dl1.h5", os.path.join(d, f"cta_{name}.dl1.h5")
            )
        yield d


@pytest.fixture
def energy_model(temp_dir):
    from aict_tools.scripts.train_energy_regressor import main as train

    model = os.path.join(temp_dir, "energy.pkl")

    runner = CliRunner()
    result = runner.invoke(
        train,
        [
            "examples/config_energy.yaml",
            os.path.join(temp_dir, "gamma.hdf5"),
            os.path.join(temp_dir, "cv_energy.hdf5"),
            model,
        ],
    )
    if result.exit_code != 0:
        print(result.output)
        print_exception(*result.exc_info)
    assert result.exit_code == 0

    return model


@pytest.fixture
def cta_energy_model(temp_dir):
    from aict_tools.scripts.train_energy_regressor import main

    runner = CliRunner()
    model = os.path.join(temp_dir, "cta_energy.pkl")
    with DateNotModified("examples/cta_gammas_diffuse.dl1.h5"):
        result = runner.invoke(
            main,
            [
                "examples/cta_full_config.yaml",
                "examples/cta_gammas_diffuse.dl1.h5",
                os.path.join(temp_dir, "cta_cv_energy.hdf5"),
                model,
            ],
        )

        if result.exit_code != 0:
            print(result.output)
            print_exception(*result.exc_info)
        assert result.exit_code == 0
    return model


@pytest.fixture
def separator_model(temp_dir):
    from aict_tools.scripts.train_separation_model import main

    runner = CliRunner()
    model = os.path.join(temp_dir, "separator.pkl")
    with DateNotModified(["examples/gamma.hdf5", "examples/proton.hdf5"]):
        result = runner.invoke(
            main,
            [
                "examples/config_separator.yaml",
                "examples/gamma.hdf5",
                "examples/proton.hdf5",
                os.path.join(temp_dir, "cv_separator.hdf5"),
                model,
            ],
        )

        if result.exit_code != 0:
            print(result.output)
            print_exception(*result.exc_info)
        assert result.exit_code == 0

    return model


@pytest.fixture
def cta_separator_model(temp_dir):
    from aict_tools.scripts.train_separation_model import main

    runner = CliRunner()
    model = os.path.join(temp_dir, "cta_separator.pkl")
    with DateNotModified(
        ["examples/cta_gammas_diffuse.dl1.h5", "examples/cta_protons.dl1.h5"]
    ):
        result = runner.invoke(
            main,
            [
                "examples/cta_full_config.yaml",
                "examples/cta_gammas_diffuse.dl1.h5",
                "examples/cta_protons.dl1.h5",
                os.path.join(temp_dir, "cta_cv_separator.hdf5"),
                model,
            ],
        )

        if result.exit_code != 0:
            print(result.output)
            print_exception(*result.exc_info)
        assert result.exit_code == 0

    return model


@pytest.fixture
def disp_models(temp_dir):
    from aict_tools.scripts.train_disp_regressor import main as train

    disp_model = os.path.join(temp_dir, "disp.pkl")
    sign_model = os.path.join(temp_dir, "sign.pkl")

    with DateNotModified("examples/gamma_diffuse.hdf5"):
        runner = CliRunner()
        result = runner.invoke(
            train,
            [
                "examples/config_source.yaml",
                "examples/gamma_diffuse.hdf5",
                os.path.join(temp_dir, "cv_disp.hdf5"),
                disp_model,
                sign_model,
            ],
        )
        if result.exit_code != 0:
            print(result.output)
            print_exception(*result.exc_info)
        assert result.exit_code == 0

    return disp_model, sign_model


@pytest.fixture
def cta_disp_models(temp_dir):
    from aict_tools.scripts.train_disp_regressor import main

    runner = CliRunner()
    disp_model = os.path.join(temp_dir, "cta_disp.pkl")
    sign_model = os.path.join(temp_dir, "cta_sign.pkl")
    with DateNotModified("examples/cta_gammas_diffuse.dl1.h5"):
        result = runner.invoke(
            main,
            [
                "examples/cta_full_config.yaml",
                "examples/cta_gammas_diffuse.dl1.h5",
                os.path.join(temp_dir, "cta_cv_disp.hdf5"),
                disp_model,
                sign_model,
            ],
        )

        if result.exit_code != 0:
            print(result.output)
            print_exception(*result.exc_info)
        assert result.exit_code == 0

    return disp_model, sign_model

@pytest.fixture
def dxdy_model(temp_dir):
    from aict_tools.scripts.train_dxdy_regressor import main as train

    dxdy_model = os.path.join(temp_dir, "dxdy.pkl")

    with DateNotModified("examples/gamma_diffuse.hdf5"):
        runner = CliRunner()
        result = runner.invoke(
            train,
            [
                "examples/config_source_dxdy.yaml",
                "examples/gamma_diffuse.hdf5",
                os.path.join(temp_dir, "cv_dxdy.hdf5"),
                dxdy_model,
            ],
        )
        if result.exit_code != 0:
            print(result.output)
            print_exception(*result.exc_info)
        assert result.exit_code == 0

    return dxdy_model

@pytest.fixture
def cta_dxdy_model(temp_dir):
    from aict_tools.scripts.train_dxdy_regressor import main as train

    dxdy_model = os.path.join(temp_dir, "cta_dxdy.pkl")

    with DateNotModified("examples/cta_gammas_diffuse.dl1.h5"):
        runner = CliRunner()
        result = runner.invoke(
            train,
            [
                "examples/cta_full_config.yaml",
                "examples/cta_gammas_diffuse.dl1.h5",
                os.path.join(temp_dir, "cta_cv_dxdy.hdf5"),
                dxdy_model,
            ],
        )
        if result.exit_code != 0:
            print(result.output)
            print_exception(*result.exc_info)
        assert result.exit_code == 0

    return dxdy_model


def test_apply_regression(temp_dir, energy_model):
    from aict_tools.scripts.apply_energy_regressor import main

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "examples/config_energy.yaml",
            os.path.join(temp_dir, "gamma.hdf5"),
            energy_model,
            "--yes",
        ],
    )

    if result.exit_code != 0:
        print(result.output)
        print_exception(*result.exc_info)

    assert result.exit_code == 0


def test_apply_regression_cta(temp_dir, cta_energy_model):
    from aict_tools.scripts.apply_energy_regressor import main
    from aict_tools.io import read_data

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "examples/cta_full_config.yaml",
            os.path.join(temp_dir, "cta_gammas_diffuse.dl1.h5"),
            cta_energy_model,
            "--yes",
        ],
    )

    if result.exit_code != 0:
        print(result.output)
        print_exception(*result.exc_info)

    res_no_chunks = read_data(
        os.path.join(temp_dir, "cta_gammas_diffuse.dl1.h5"),
        "/dl2/event/telescope/tel_001/gamma_energy_prediction",
    )

    result_2 = runner.invoke(
        main,
        [
            "examples/cta_full_config.yaml",
            os.path.join(temp_dir, "cta_gammas_diffuse.dl1.h5"),
            cta_energy_model,
            "--yes",
            "-N 50",
        ],
    )

    if result_2.exit_code != 0:
        print(result_2.output)
        print_exception(*result_2.exc_info)

    res_chunks = read_data(
        os.path.join(temp_dir, "cta_gammas_diffuse.dl1.h5"),
        "/dl2/event/telescope/tel_001/gamma_energy_prediction",
    )

    assert (res_no_chunks.all() == res_chunks.all()).all()


def test_apply_separator(temp_dir, separator_model):
    from aict_tools.scripts.apply_separation_model import main as apply_model
    import h5py

    runner = CliRunner()
    result = runner.invoke(
        apply_model,
        [
            "examples/config_separator.yaml",
            os.path.join(temp_dir, "gamma.hdf5"),
            separator_model,
            "--yes",
        ],
    )

    if result.exit_code != 0:
        print(result.output)
        print_exception(*result.exc_info)
    assert result.exit_code == 0

    with h5py.File(os.path.join(temp_dir, "gamma.hdf5"), "r") as f:
        assert "gammaness" in f["events"]


def test_apply_separator_cta(temp_dir, cta_separator_model):
    from aict_tools.scripts.apply_separation_model import main as apply_model
    import h5py

    runner = CliRunner()
    result = runner.invoke(
        apply_model,
        [
            "examples/cta_full_config.yaml",
            os.path.join(temp_dir, "cta_gammas.dl1.h5"),
            cta_separator_model,
            "--yes",
        ],
    )

    if result.exit_code != 0:
        print(result.output)
        print_exception(*result.exc_info)
    assert result.exit_code == 0

    with h5py.File(os.path.join(temp_dir, "cta_gammas.dl1.h5"), "r") as f:
        assert "gamma_prediction" in f["dl2"]["event"]["telescope"]["tel_001"]
        assert "gamma_prediction" in f["dl2"]["event"]["subarray"]


def test_train_disp_altitude():
    from aict_tools.scripts.train_disp_regressor import main as train

    with tempfile.TemporaryDirectory(prefix="aict_tools_test_") as d:

        with DateNotModified("examples/gamma_diffuse_altitude.hdf5"):
            runner = CliRunner()
            result = runner.invoke(
                train,
                [
                    "examples/config_source_altitude.yaml",
                    "examples/gamma_diffuse_altitude.hdf5",
                    os.path.join(d, "test.hdf5"),
                    os.path.join(d, "disp.pkl"),
                    os.path.join(d, "sign.pkl"),
                ],
            )
            if result.exit_code != 0:
                print(result.output)
                print_exception(*result.exc_info)
            assert result.exit_code == 0


def test_apply_disp(temp_dir, disp_models):
    from aict_tools.scripts.apply_disp_regressor import main as apply_model

    disp_model, sign_model = disp_models

    runner = CliRunner()
    result = runner.invoke(
        apply_model,
        [
            "examples/config_source.yaml",
            os.path.join(temp_dir, "gamma.hdf5"),
            disp_model,
            sign_model,
            "--yes",
        ],
    )

    if result.exit_code != 0:
        print(result.output)
        print_exception(*result.exc_info)
    assert result.exit_code == 0
    with h5py.File(os.path.join(temp_dir, "gamma.hdf5"), "r") as f:
        assert "source_x_prediction" in f["events"]


def test_apply_dxdy(temp_dir, dxdy_model):
    from aict_tools.scripts.apply_dxdy_regressor import main as apply_model

    runner = CliRunner()
    result = runner.invoke(
        apply_model,
        [
            "examples/config_source_dxdy.yaml",
            os.path.join(temp_dir, "gamma.hdf5"),
            dxdy_model,
            "--yes",
        ],
    )

    if result.exit_code != 0:
        print(result.output)
        print_exception(*result.exc_info)
    assert result.exit_code == 0


def test_apply_disp_cta(temp_dir, cta_disp_models):
    from aict_tools.scripts.apply_disp_regressor import main as apply_model

    cta_disp_model, cta_sign_model = cta_disp_models
    runner = CliRunner()
    result = runner.invoke(
        apply_model,
        [
            "examples/cta_full_config.yaml",
            os.path.join(temp_dir, "cta_gammas.dl1.h5"),
            cta_disp_model,
            cta_sign_model,
            "--yes",
        ],
    )

    if result.exit_code != 0:
        print(result.output)
        print_exception(*result.exc_info)
    assert result.exit_code == 0

    with h5py.File(os.path.join(temp_dir, "cta_gammas.dl1.h5"), "r") as f:
        assert "disp_prediction" in f["dl2"]["event"]["telescope"]["tel_001"]


def test_apply_dxdy_cta(temp_dir, cta_dxdy_model):
    from aict_tools.scripts.apply_dxdy_regressor import main as apply_model

    runner = CliRunner()
    result = runner.invoke(
        apply_model,
        [
            "examples/cta_full_config.yaml",
            os.path.join(temp_dir, "cta_gammas.dl1.h5"),
            cta_dxdy_model,
            "--yes",
        ],
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

    with tempfile.TemporaryDirectory(prefix="aict_tools_test_") as d:

        runner = CliRunner()

        with DateNotModified(
            [
                "examples/crab.hdf5",
                "examples/gamma_diffuse.hdf5",
                "examples/gamma.hdf5",
                "examples/proton.hdf5",
            ]
        ):

            result = runner.invoke(
                train_disp,
                [
                    "examples/full_config.yaml",
                    "examples/gamma_diffuse.hdf5",
                    os.path.join(d, "disp_performance.hdf5"),
                    os.path.join(d, "disp.pkl"),
                    os.path.join(d, "sign.pkl"),
                ],
            )
            if result.exit_code != 0:
                print(result.output)
                print_exception(*result.exc_info)
            assert result.exit_code == 0

            result = runner.invoke(
                train_energy,
                [
                    "examples/full_config.yaml",
                    "examples/gamma.hdf5",
                    os.path.join(d, "regressor_performance.hdf5"),
                    os.path.join(d, "regressor.pkl"),
                ],
            )

            if result.exit_code != 0:
                print(result.output)
                print_exception(*result.exc_info)
            assert result.exit_code == 0

            result = runner.invoke(
                train_separator,
                [
                    "examples/full_config.yaml",
                    "examples/gamma.hdf5",
                    "examples/proton.hdf5",
                    os.path.join(d, "separator_performance.hdf5"),
                    os.path.join(d, "separator.pkl"),
                ],
            )

            if result.exit_code != 0:
                print(result.output)
                print_exception(*result.exc_info)
            assert result.exit_code == 0

            result = runner.invoke(
                to_dl3,
                [
                    "examples/full_config.yaml",
                    "examples/crab.hdf5",
                    os.path.join(d, "separator.pkl"),
                    os.path.join(d, "regressor.pkl"),
                    os.path.join(d, "disp.pkl"),
                    os.path.join(d, "sign.pkl"),
                    os.path.join(d, "crab_dl3.hdf5"),
                ],
            )

            if result.exit_code != 0:
                print(result.output)
                print_exception(*result.exc_info)
            assert result.exit_code == 0

            output = os.path.join(d, "gamma_dl3.hdf5")
            result = runner.invoke(
                to_dl3,
                [
                    "examples/full_config.yaml",
                    "examples/gamma.hdf5",
                    os.path.join(d, "separator.pkl"),
                    os.path.join(d, "regressor.pkl"),
                    os.path.join(d, "disp.pkl"),
                    os.path.join(d, "sign.pkl"),
                    output,
                ],
            )

            if result.exit_code != 0:
                print(result.output)
                print_exception(*result.exc_info)
            assert result.exit_code == 0

            with h5py.File(output, "r") as f:
                assert f.attrs["sample_fraction"] == 1000 / 1851297


def test_split_data_executable():
    from aict_tools.scripts.split_data import main as split

    with tempfile.TemporaryDirectory(prefix="aict_tools_test_") as d:

        infile = os.path.join(d, "gamma.hdf5")
        shutil.copy("examples/gamma.hdf5", infile)
        with DateNotModified(infile):

            runner = CliRunner()
            result = runner.invoke(
                split,
                [
                    infile,
                    os.path.join(d, "signal"),
                    "-ntest",  # no spaces here. maybe a bug in click?
                    "-f0.75",
                    "-ntrain",
                    "-f0.25",
                ],
            )
            if result.exit_code != 0:
                print(result.output)
                print_exception(*result.exc_info)
            assert result.exit_code == 0

            print(os.listdir(d))
            test_path = os.path.join(d, "signal_test.hdf5")
            assert os.path.isfile(test_path)

            with h5py.File(test_path, "r") as f:
                assert f.attrs["sample_fraction"] == 0.75

            train_path = os.path.join(d, "signal_train.hdf5")
            assert os.path.isfile(train_path)

            with h5py.File(train_path, "r") as f:
                assert f.attrs["sample_fraction"] == 0.25


def test_split_data_executable_cta():
    from aict_tools.scripts.split_data import main as split

    with tempfile.TemporaryDirectory(prefix="aict_tools_test_") as d:

        infile = os.path.join(d, "cta_gammas_diffuse.dl1.h5")
        shutil.copy("examples/cta_gammas_diffuse.dl1.h5", infile)
        with DateNotModified(infile):

            runner = CliRunner()
            result = runner.invoke(
                split,
                [
                    infile,
                    os.path.join(d, "signal"),
                    "-ntest",  # no spaces here. maybe a bug in click?
                    "-f0.75",
                    "-ntrain",
                    "-f0.25",
                    "-dCTA",
                ],
            )
            if result.exit_code != 0:
                print(result.output)
                print_exception(*result.exc_info)
            assert result.exit_code == 0

            test_path = os.path.join(d, "signal_test.dl1.h5")
            assert os.path.isfile(test_path)

            with h5py.File(test_path, "r") as f:
                # tolerance because event sample fraction need to be rounded with
                # n_events = 1100
                assert np.isclose(f.attrs["sample_fraction"], 0.75, atol=0.005)

            train_path = os.path.join(d, "signal_train.dl1.h5")
            assert os.path.isfile(train_path)

            with h5py.File(train_path, "r") as f:
                assert np.isclose(f.attrs["sample_fraction"], 0.25, atol=0.005)


def test_split_data_executable_chunked():
    from aict_tools.scripts.split_data import main as split

    with tempfile.TemporaryDirectory(prefix="aict_tools_test_") as d:

        infile = os.path.join(d, "gamma.hdf5")
        shutil.copy("examples/gamma.hdf5", infile)
        with DateNotModified(infile):

            runner = CliRunner()
            result = runner.invoke(
                split,
                [
                    infile,
                    os.path.join(d, "signal"),
                    "-ntest",  # no spaces here. maybe a bug in click?
                    "-f0.75",
                    "-ntrain",
                    "-f0.25",
                    "--chunksize=100",
                ],
            )
            if result.exit_code != 0:
                print(result.output)
                print_exception(*result.exc_info)
            assert result.exit_code == 0

            print(os.listdir(d))
            test_path = os.path.join(d, "signal_test.hdf5")
            assert os.path.isfile(test_path)

            with h5py.File(test_path, "r") as f:
                assert f.attrs["sample_fraction"] == 0.75

            train_path = os.path.join(d, "signal_train.hdf5")
            assert os.path.isfile(train_path)

            with h5py.File(train_path, "r") as f:
                assert f.attrs["sample_fraction"] == 0.25


def test_apply_regression_pmml():
    importorskip("jpmml_evaluator")
    importorskip("sklearn2pmml")

    from aict_tools.scripts.train_energy_regressor import main as train
    from aict_tools.scripts.apply_energy_regressor import main as apply

    with tempfile.TemporaryDirectory(prefix="aict_tools_test_") as d:
        runner = CliRunner()

        shutil.copy("examples/gamma.hdf5", os.path.join(d, "gamma.hdf5"))

        result = runner.invoke(
            train,
            [
                "examples/config_energy.yaml",
                os.path.join(d, "gamma.hdf5"),
                os.path.join(d, "test.hdf5"),
                os.path.join(d, "test.pmml"),
            ],
        )
        if result.exit_code != 0:
            print(result.output)
            print_exception(*result.exc_info)
        assert result.exit_code == 0

        result = runner.invoke(
            apply,
            [
                "examples/config_energy.yaml",
                os.path.join(d, "gamma.hdf5"),
                os.path.join(d, "test.pmml"),
                "--yes",
            ],
        )

        if result.exit_code != 0:
            print(result.output)
            print_exception(*result.exc_info)

        assert result.exit_code == 0


def test_apply_separator_pmml():
    importorskip("jpmml_evaluator")
    importorskip("sklearn2pmml")

    from aict_tools.scripts.train_separation_model import main as train
    from aict_tools.scripts.apply_separation_model import main as apply_model
    import h5py

    with tempfile.TemporaryDirectory(prefix="aict_tools_test_") as d:
        shutil.copy("examples/gamma.hdf5", os.path.join(d, "gamma.hdf5"))

        runner = CliRunner()
        result = runner.invoke(
            train,
            [
                "examples/config_separator.yaml",
                "examples/gamma.hdf5",
                "examples/proton.hdf5",
                os.path.join(d, "test.hdf5"),
                os.path.join(d, "test.pmml"),
            ],
        )
        if result.exit_code != 0:
            print(result.output)
            print_exception(*result.exc_info)
        assert result.exit_code == 0

        result = runner.invoke(
            apply_model,
            [
                "examples/config_separator.yaml",
                os.path.join(d, "gamma.hdf5"),
                os.path.join(d, "test.pmml"),
                "--yes",
            ],
        )

        if result.exit_code != 0:
            print(result.output)
            print_exception(*result.exc_info)
        assert result.exit_code == 0

        with h5py.File(os.path.join(d, "gamma.hdf5"), "r") as f:
            assert "gammaness" in f["events"]


def test_apply_regression_onnx():
    importorskip("onnxruntime")
    importorskip("skl2onnx")

    from aict_tools.scripts.train_energy_regressor import main as train
    from aict_tools.scripts.apply_energy_regressor import main as apply

    with tempfile.TemporaryDirectory(prefix="aict_tools_test_") as d:
        runner = CliRunner()

        shutil.copy("examples/gamma.hdf5", os.path.join(d, "gamma.hdf5"))

        result = runner.invoke(
            train,
            [
                "examples/config_energy.yaml",
                os.path.join(d, "gamma.hdf5"),
                os.path.join(d, "test.hdf5"),
                os.path.join(d, "test.onnx"),
            ],
        )
        if result.exit_code != 0:
            print(result.output)
            print_exception(*result.exc_info)
        assert result.exit_code == 0

        result = runner.invoke(
            apply,
            [
                "examples/config_energy.yaml",
                os.path.join(d, "gamma.hdf5"),
                os.path.join(d, "test.onnx"),
                "--yes",
            ],
        )

        if result.exit_code != 0:
            print(result.output)
            print_exception(*result.exc_info)

        assert result.exit_code == 0


def test_apply_separator_onnx():
    importorskip("onnxruntime")
    importorskip("skl2onnx")
    from aict_tools.scripts.train_separation_model import main as train
    from aict_tools.scripts.apply_separation_model import main as apply_model
    import h5py

    with tempfile.TemporaryDirectory(prefix="aict_tools_test_") as d:
        shutil.copy("examples/gamma.hdf5", os.path.join(d, "gamma.hdf5"))

        runner = CliRunner()
        result = runner.invoke(
            train,
            [
                "examples/config_separator.yaml",
                "examples/gamma.hdf5",
                "examples/proton.hdf5",
                os.path.join(d, "test.hdf5"),
                os.path.join(d, "test.onnx"),
            ],
        )
        if result.exit_code != 0:
            print(result.output)
            print_exception(*result.exc_info)
        assert result.exit_code == 0

        result = runner.invoke(
            apply_model,
            [
                "examples/config_separator.yaml",
                os.path.join(d, "gamma.hdf5"),
                os.path.join(d, "test.onnx"),
                "--yes",
            ],
        )

        if result.exit_code != 0:
            print(result.output)
            print_exception(*result.exc_info)
        assert result.exit_code == 0

        with h5py.File(os.path.join(d, "gamma.hdf5"), "r") as f:
            assert "gammaness" in f["events"]


def test_plot_separator_performance(temp_dir, separator_model):
    from aict_tools.scripts.plot_separator_performance import main

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "examples/config_separator.yaml",
            os.path.join(temp_dir, "cv_separator.hdf5"),
            separator_model,
            "-o",
            os.path.join(temp_dir, "separator_performance.pdf"),
        ],
    )
    if result.exit_code != 0:
        print(result.output)
        print_exception(*result.exc_info)
    assert result.exit_code == 0


def test_plot_regressor_performance(temp_dir, energy_model):
    from aict_tools.scripts.plot_regressor_performance import main

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "examples/config_energy.yaml",
            os.path.join(temp_dir, "cv_energy.hdf5"),
            energy_model,
            "-o",
            os.path.join(temp_dir, "energy_performance.pdf"),
        ],
    )
    if result.exit_code != 0:
        print(result.output)
        print_exception(*result.exc_info)
    assert result.exit_code == 0


def test_plot_disp_performance(temp_dir, disp_models):
    from aict_tools.scripts.plot_disp_performance import main

    disp_model, sign_model = disp_models

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "examples/config_source.yaml",
            os.path.join(temp_dir, "cv_disp.hdf5"),
            "examples/gamma_diffuse.hdf5",
            disp_model,
            sign_model,
            "-o",
            os.path.join(temp_dir, "disp_performance.pdf"),
        ],
    )
    if result.exit_code != 0:
        print(result.output)
        print_exception(*result.exc_info)
    assert result.exit_code == 0
