import click
import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.externals import joblib
from ..configuration import AICTConfig
import fact.io

from ..plotting import (
    plot_roc,
    plot_probabilities,
    plot_precision_recall,
    plot_feature_importances,
)


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('performance_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('model_path', type=click.Path(exists=True, dir_okay=False))
@click.option('-o', '--output', type=click.Path(exists=False, dir_okay=False))
@click.option('-k', '--key', help='HDF5 key for pandas hdf5', default='data')
def main(configuration_path, performance_path, model_path, output, key):
    ''' Create some performance evaluation plots for the separator '''

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger()

    log.info('Loading perfomance data')
    df = fact.io.read_data(performance_path, key=key)

    log.info('Loading model')
    model = joblib.load(model_path)

    model_config = AICTConfig.from_yaml(configuration_path).separator

    log.info('Creating performance plots. ')
    figures = []

    # Plot rocs
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    plot_roc(df, model, ax=ax)


    # Plot hists of probas
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)

    plot_probabilities(df, model, ax=ax)

    # Plot hists of probas
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)

    plot_precision_recall(df, model, ax=ax)

    # Plot feature importances
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)

    features = model_config.features

    plot_feature_importances(model, features, ax=ax)

    if output is None:
        plt.show()
    else:
        with PdfPages(output) as pdf:
            for fig in figures:
                pdf.savefig(fig)
