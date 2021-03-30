import click
import logging
import matplotlib
import matplotlib.pyplot as plt
import joblib
import fact.io

from ..configuration import AICTConfig
from ..plotting import (
    plot_regressor_confusion,
    plot_feature_importances,
    plot_true_delta_delta,
    plot_energy_dependent_dxdy_metrics,
)
from ..io import read_telescope_data

if matplotlib.get_backend() == "pgf":
    from matplotlib.backends.backend_pgf import PdfPages
else:
    from matplotlib.backends.backend_pdf import PdfPages


@click.command()
@click.argument("configuration_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("performance_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("data_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("dxdy_model_path", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", type=click.Path(exists=False, dir_okay=False))
@click.option(
    "-k", "--key", help="HDF5 key for hdf5 for performance_path", default="data"
)
@click.option(
    "-k_data", "--key_data", help="HDF5 key for hdf5 for data_path", default="events"
)
def main(
    configuration_path,
    performance_path,
    data_path,
    dxdy_model_path,
    output,
    key,
    key_data,
):
    """ Create some performance evaluation plots for the dxdy model"""
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger()

    config = AICTConfig.from_yaml(configuration_path)
    model_config = config.dxdy

    log.info("Loading perfomance data")
    df = fact.io.read_data(performance_path, key=key)

    columns = model_config.columns_to_read_train

    if model_config.coordinate_transformation == "CTA":
        camera_unit = r"\mathrm{m}"
    else:
        camera_unit = r"\mathrm{mm}"

    log.info('Loading original data')
    df_data = read_telescope_data(
        data_path,
        config,
        model_config.columns_to_read_train,
    )

    log.info("Loading dxdy model")
    dxdy_model = joblib.load(dxdy_model_path)

    figures = []

    # Plot confusion dx log
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    ax.set_title("Reconstructed vs. True dx (log color scale)")
    plot_regressor_confusion(
        df, log_xy=False, ax=ax, label_column="dx", prediction_column="dx_prediction"
    )
    ax.set_xlabel(r"$dx_{\mathrm{MC}} \,\, / \,\, " + camera_unit + "$")
    ax.set_ylabel(r"$dx_{\mathrm{Est}} \,\, / \,\, " + camera_unit + "$")

    # Plot confusion dx linear
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    ax.set_title("Reconstructed vs. True dx (linear color scale)")
    plot_regressor_confusion(
        df,
        log_xy=False,
        log_z=False,
        ax=ax,
        label_column="dx",
        prediction_column="dx_prediction",
    )
    ax.set_xlabel(r"$dx_{\mathrm{MC}} \,\, / \,\, " + camera_unit + "$")
    ax.set_ylabel(r"$dx_{\mathrm{Est}} \,\, / \,\, " + camera_unit + "$")

    # Plot confusion dy log
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    # ax.set_title('Reconstructed vs. True dy (log color scale)')
    ax.set_title("Rekonstruiertes vs. wahres dy")
    plot_regressor_confusion(
        df, log_xy=False, ax=ax, label_column="dy", prediction_column="dy_prediction"
    )
    ax.set_xlabel(r"$dy_{\mathrm{MC}} \,\, / \,\, " + camera_unit + "$")
    ax.set_ylabel(r"$dy_{\mathrm{Est}} \,\, / \,\, " + camera_unit + "$")

    # Plot confusion dy linear
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    ax.set_title("Reconstructed vs. True dy (linear color scale)")
    plot_regressor_confusion(
        df,
        log_xy=False,
        log_z=False,
        ax=ax,
        label_column="dy",
        prediction_column="dy_prediction",
    )
    ax.set_xlabel(r"$dy_{\mathrm{MC}} \,\, / \,\, " + camera_unit + "$")
    ax.set_ylabel(r"$dy_{\mathrm{Est}} \,\, / \,\, " + camera_unit + "$")

    # Plot feature importances dxdy
    if hasattr(dxdy_model, "feature_importances_"):
        figures.append(plt.figure())
        ax = figures[-1].add_subplot(1, 1, 1)
        ax.set_title(r"Feature Importance dxdy")

        features = model_config.features
        plot_feature_importances(dxdy_model, features, ax=ax)

    # Plot true_delta - delta
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    plot_true_delta_delta(df_data, model_config, ax)

    if config.true_energy_column in df.columns:
        fig = plot_energy_dependent_dxdy_metrics(
            df, config.true_energy_column, energy_unit=config.energy_unit
        )
        figures.append(fig)

    if output is None:
        plt.show()
    else:
        with PdfPages(output) as pdf:
            for fig in figures:
                fig.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
                pdf.savefig(fig)


if __name__ == "__main__":
    main()
