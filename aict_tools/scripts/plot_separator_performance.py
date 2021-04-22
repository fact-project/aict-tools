import click
import logging
import matplotlib.pyplot as plt
import matplotlib
import joblib
import fact.io
import numpy as np

from ..plotting import (
    plot_roc,
    plot_scores,
    plot_precision_recall,
    plot_feature_importances,
    plot_rocauc_vs_size,
)
from ..configuration import AICTConfig

if matplotlib.get_backend() == "pgf":
    from matplotlib.backends.backend_pgf import PdfPages
else:
    from matplotlib.backends.backend_pdf import PdfPages


@click.command()
@click.argument("configuration_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("performance_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", type=click.Path(exists=False, dir_okay=False))
@click.option("-k", "--key", help="HDF5 key for pandas hdf5", default="data")
def main(configuration_path, performance_path, model_path, output, key):
    """ Create some performance evaluation plots for the separator """

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger()

    log.info("Loading perfomance data")
    df = fact.io.read_data(performance_path, key=key)

    log.info("Loading model")
    model = joblib.load(model_path)

    config = AICTConfig.from_yaml(configuration_path)
    model_config = config.separator

    log.info("Creating performance plots. ")
    figures = []

    # Plot rocs
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    plot_roc(df, model, score_column=model_config.output_name, ax=ax)

    # Plot hists of probas
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)

    plot_scores(
        df,
        model,
        score_column=model_config.output_name,
        ax=ax,
        xlabel=model_config.output_name,
    )

    # Plot hists of probas
    if config.size_column is not None:
        figures.append(plt.figure())
        ax = figures[-1].add_subplot(1, 1, 1)

        median = np.nanmedian(df[config.size_column])
        mask = df[config.size_column] > median

        ax.set_title(f'Scores for {config.size_column} > {median:.1f} (median value)')
        plot_scores(
            df[mask],
            model,
            score_column=model_config.output_name,
            ax=ax,
            xlabel=model_config.output_name,
        )

        figures.append(plt.figure())
        ax = figures[-1].add_subplot(1, 1, 1)
        mask = df[config.size_column] <= median

        ax.set_title(f'Scores for {config.size_column} <= {median:.1f} (median value)')
        plot_scores(
            df[mask],
            model,
            score_column=model_config.output_name,
            ax=ax,
            xlabel=model_config.output_name,
        )

    # Plot hists of probas
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)

    plot_precision_recall(df, model, ax=ax, score_column=model_config.output_name)

    # Plot feature importances
    if hasattr(model, "feature_importances_"):
        figures.append(plt.figure())
        ax = figures[-1].add_subplot(1, 1, 1)

        features = model_config.features
        plot_feature_importances(model, features, ax=ax)

    if config.size_column is not None:
        figures.append(plt.figure())
        ax = figures[-1].add_subplot(1, 1, 1)
        plot_rocauc_vs_size(df, config.size_column, ax=ax, score_column=model_config.output_name)


    if output is None:
        plt.show()
    else:
        with PdfPages(output) as pdf:
            for fig in figures:
                fig.tight_layout(pad=0)
                pdf.savefig(fig)
