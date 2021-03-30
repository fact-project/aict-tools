import click
import numpy as np
from tqdm import tqdm

from ..io import (
    append_column_to_hdf5,
    append_predictions_cta,
    read_telescope_data_chunked,
    drop_prediction_column,
    drop_prediction_groups,
    load_model,
)
from ..apply import predict_disp
from ..configuration import AICTConfig
from ..logging import setup_logging
from ..preprocessing import convert_units


@click.command()
@click.argument("configuration_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("data_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("disp_model_path", type=click.Path(exists=False, dir_okay=False))
@click.argument("sign_model_path", type=click.Path(exists=False, dir_okay=False))
@click.option("-n", "--n-jobs", type=int, help="Number of cores to use")
@click.option("-y", "--yes", help="Do not prompt for overwrites", is_flag=True)
@click.option("-v", "--verbose", help="Verbose log output", is_flag=True)
@click.option(
    "-N",
    "--chunksize",
    type=int,
    help="If given, only process the given number of events at once",
)
def main(
    configuration_path,
    data_path,
    disp_model_path,
    sign_model_path,
    chunksize,
    n_jobs,
    yes,
    verbose,
):
    """
    Apply given model to data. Three columns are added to the file:
    source_x_prediction, source_y_prediction and disp_prediction

    CONFIGURATION_PATH: Path to the config yaml file
    DATA_PATH: path to the FACT data in a h5py hdf5 file, e.g. erna_gather_fits output
    DISP_MODEL_PATH: Path to the pickled disp model.
    SIGN_MODEL_PATH: Path to the pickled sign model.
    """
    log = setup_logging(verbose=verbose)

    config = AICTConfig.from_yaml(configuration_path)
    model_config = config.disp

    columns_to_delete = [
        "source_x_prediction",
        "source_y_prediction",
        "theta",
        "theta_deg",
        "theta_rec_pos",
        "disp_prediction",
    ]
    for i in range(1, 6):
        columns_to_delete.extend(
            [
                "theta_off_" + str(i),
                "theta_deg_off_" + str(i),
                "theta_off_rec_pos_" + str(i),
            ]
        )

    n_del_cols = 0

    for column in columns_to_delete:
        if config.data_format == "CTA":
            n_del = drop_prediction_groups(data_path, group_name=column, yes=yes)
        elif config.data_format == "simple":
            n_del = drop_prediction_column(
                data_path, group_name=config.events_key, column_name=column, yes=yes
            )
        n_del_cols += n_del

    if config.data_format == "CTA":
        drop_prediction_groups(data_path, group_name=model_config.output_name, yes=yes)

    if n_del_cols > 0:
        log.warning(
            "Source dependent features need to be calculated from the predicted source possition. "
            + "Use e.g. `fact_calculate_theta` from https://github.com/fact-project/pyfact."
        )

    log.info("Loading model")
    disp_model = load_model(disp_model_path)
    sign_model = load_model(sign_model_path)
    log.info("Done")

    if n_jobs:
        disp_model.n_jobs = n_jobs
        sign_model.n_jobs = n_jobs

    df_generator = read_telescope_data_chunked(
        data_path,
        config,
        chunksize,
        model_config.columns_to_read_apply,
        feature_generation_config=model_config.feature_generation,
    )

    log.info("Predicting on data...")
    for df_data, start, stop in tqdm(df_generator):
        disp = predict_disp(
            df_data[model_config.features],
            disp_model,
            sign_model,
            log_target=model_config.log_target,
        )

        source_x = df_data[config.cog_x_column].values + disp * np.cos(
            df_data[config.delta_column].values
        )
        source_y = df_data[config.cog_y_column].values + disp * np.sin(
            df_data[config.delta_column].values
        )

        if config.data_format == "CTA":
            df_data.reset_index(inplace=True)
            for tel_id, group in df_data.groupby("tel_id"):
                d = group[["obs_id", "event_id"]].copy()
                d["source_y_pred"] = source_y[group.index]
                d["source_x_pred"] = source_x[group.index]
                d["disp_pred"] = disp[group.index]
                append_predictions_cta(
                    data_path,
                    d,
                    f"/dl2/event/telescope/tel_{tel_id:03d}",
                    model_config.output_name,
                )

        elif config.data_format == "simple":
            key = config.events_key
            append_column_to_hdf5(data_path, source_x, key, "source_x_prediction")
            append_column_to_hdf5(data_path, source_y, key, "source_y_prediction")
            append_column_to_hdf5(data_path, disp, key, "disp_prediction")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
