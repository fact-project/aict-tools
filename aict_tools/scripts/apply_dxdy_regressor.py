import click
import numpy as np
from tqdm import tqdm

from ..io import (
    append_column_to_hdf5,
    append_predictions_cta,
    append_predictions_cta,
    drop_prediction_column,
    drop_prediction_groups,
    load_model,
    read_telescope_data_chunked,
)
from ..apply import predict_dxdy
from ..configuration import AICTConfig
from ..logging import setup_logging
from ..preprocessing import calc_true_disp


@click.command()
@click.argument("configuration_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("data_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("dxdy_model_path", type=click.Path(exists=False, dir_okay=False))
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
    configuration_path, data_path, dxdy_model_path, chunksize, n_jobs, yes, verbose
):
    """
    Apply given model to data. Three columns are added to the file, source_x_prediction, source_y_prediction
    and disp_prediction

    CONFIGURATION_PATH: Path to the config yaml file
    DATA_PATH: path to the FACT data in a h5py hdf5 file, e.g. erna_gather_fits output
    DXDY_MODEL_PATH: Path to the pickled dxdy model.
    """
    log = setup_logging(verbose=verbose)

    config = AICTConfig.from_yaml(configuration_path)
    model_config = config.dxdy

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
        n_del = 0
        if config.data_format == "CTA":
            n_del = drop_prediction_groups(data_path, group_name=column, yes=yes)
        elif config.data_format == "simple":
            n_del = drop_prediction_column(
                data_path, group_name=config.events_key, column_name=column, yes=yes
            )
        n_del_cols += n_del

    if n_del_cols > 0:
        log.warn(
            "Source dependent features need to be calculated from the predicted source possition. "
            + "Use e.g. `fact_calculate_theta` from https://github.com/fact-project/pyfact."
        )

    log.info("Loading model")
    dxdy_model = load_model(dxdy_model_path)
    log.info("Done")

    if n_jobs:
        dxdy_model.n_jobs = n_jobs

    df_generator = read_telescope_data_chunked(
        data_path,
        config,
        chunksize,
        model_config.columns_to_read_apply,
        feature_generation_config=model_config.feature_generation,
    )

    log.info("Predicting on data...")
    for df_data, start, stop in tqdm(df_generator):

        dxdy = predict_dxdy(
            df_data[model_config.features],
            dxdy_model,
        )

        source_x = df_data[config.cog_x_column] + dxdy[:, 0]
        source_y = df_data[config.cog_y_column] + dxdy[:, 1]
        if config.data_format == "CTA":
            df_data.reset_index(inplace=True)
            for tel_id, group in df_data.groupby("tel_id"):
                d = group[["obs_id", "event_id"]].copy()
                d["source_y_prediction"] = source_y[group.index]
                d["source_x_prediction"] = source_x[group.index]
                d["dx_prediction"] = dxdy[:, 0][group.index]
                d["dy_prediction"] = dxdy[:, 1][group.index]
                append_predictions_cta(
                    data_path,
                    d,
                    f"/dl2/event/telescope/{model_config.output_name}/tel_{tel_id:03d}",
                )

        elif config.data_format == "simple":
            key = config.events_key
            append_column_to_hdf5(data_path, source_x, key, "source_x_prediction")
            append_column_to_hdf5(data_path, source_y, key, "source_y_prediction")
            append_column_to_hdf5(data_path, dxdy[:, 0], key, "dx_prediction")
            append_column_to_hdf5(data_path, dxdy[:, 1], key, "dy_prediction")

if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
