import click
import logging
import matplotlib
import matplotlib.pyplot as plt
import joblib
import fact.io

from ..io import read_telescope_data
from ..configuration import AICTConfig
from ..plotting import (
    plot_roc,
    plot_scores,
    plot_regressor_confusion,
    plot_feature_importances,
    plot_true_delta_delta,
    plot_energy_dependent_disp_metrics,
)

if matplotlib.get_backend() == "pgf":
    from matplotlib.backends.backend_pgf import PdfPages
else:
    from matplotlib.backends.backend_pdf import PdfPages


@click.command()
@click.argument("configuration_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("performance_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("data_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("disp_model_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("sign_model_path", type=click.Path(exists=True, dir_okay=False))
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
    disp_model_path,
    sign_model_path,
    output,
    key,
    key_data,
):
    """ Create some performance evaluation plots for the disp model"""
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger()

    config = AICTConfig.from_yaml(configuration_path)
    model_config = config.disp

    log.info("Loading perfomance data")

    df = fact.io.read_data(performance_path, key=key)

    if config.data_format == "CTA":
        camera_unit = r"\mathrm{m}"
    else:
        camera_unit = r"\mathrm{mm}"

    log.info("Loading original data")
    df_data = read_telescope_data(
        data_path,
        config,
        model_config.columns_to_read_train,
    )

    log.info("Loading disp model")
    disp_model = joblib.load(disp_model_path)

    log.info("Loading sign model")
    sign_model = joblib.load(sign_model_path)

    figures = []

    # Plot confusion
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    ax.set_title("Reconstructed vs. True |disp| (log color scale)")
    plot_regressor_confusion(
        df,
        log_xy=False,
        ax=ax,
        label_column="disp",
        prediction_column="disp_prediction",
    )
    ax.set_xlabel(r"$|disp|_{\mathrm{MC}} \,\, / \,\, " + camera_unit + "$")
    ax.set_ylabel(r"$|disp|_{\mathrm{Est}} \,\, / \,\, " + camera_unit + "$")

    # Plot confusion
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    ax.set_title("Reconstructed vs. True |disp| (linear color scale)")
    plot_regressor_confusion(
        df,
        log_xy=False,
        log_z=False,
        ax=ax,
        label_column="disp",
        prediction_column="disp_prediction",
    )
    ax.set_xlabel(r"$|disp|_{\mathrm{MC}} \,\, / \,\, " + camera_unit + "$")
    ax.set_ylabel(r"$|disp|_{\mathrm{Est}} \,\, / \,\, " + camera_unit + "$")

    # Plot ROC
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    plot_roc(df, sign_model, ax=ax, label_column="sign", score_column="sign_score")

    # Plot hists of probas
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    plot_scores(
        df,
        sign_model,
        ax=ax,
        classnames={-1.0: r"$-$", 1.0: r"$+$"},
        label_column="sign",
        score_column="sign_score",
    )

    # Plot feature importances sign
    if hasattr(sign_model, "feature_importances_"):
        figures.append(plt.figure())
        ax = figures[-1].add_subplot(1, 1, 1)
        ax.set_title("Feature Importance sign")

        features = model_config.features

        plot_feature_importances(sign_model, features, ax=ax)

    # Plot feature importances disp
    if hasattr(disp_model, "feature_importances_"):
        figures.append(plt.figure())
        ax = figures[-1].add_subplot(1, 1, 1)
        ax.set_title(r"Feature Importance |disp|")

        features = model_config.features
        plot_feature_importances(disp_model, features, ax=ax)

    # Plot true_delta - delta
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    plot_true_delta_delta(df_data, config, ax)

    if config.true_energy_column in df.columns:
        fig = plot_energy_dependent_disp_metrics(
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
