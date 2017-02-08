import click
import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.externals import joblib
import yaml
import pandas as pd

from ..plotting import plot_roc, plot_probabilities, plot_precision_recall


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('performance_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('model_path', type=click.Path(exists=True, dir_okay=False))
@click.option('-o', '--output', type=click.Path(exists=False, dir_okay=False))
@click.option('-k', '--key', help='HDF5 key for pandas hdf5', default='table')
def main(configuration_path, performance_path, model_path, output, key):
    ''' Create some performance evaluation plots for the separator '''

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger()

    log.info('Loading perfomance data')
    df = pd.read_hdf(performance_path, key)

    log.info('Loading model')
    model = joblib.load(model_path)

    with open(configuration_path) as f:
        config = yaml.load(f)

    # Plot rocs
    fig_roc = plt.figure()
    ax_roc = fig_roc.add_subplot(1, 1, 1)
    plot_roc(df, model, ax=ax_roc)

    # Plot hists of probas
    fig_probas = plt.figure()
    ax_probas = fig_probas.add_subplot(1, 1, 1)

    plot_probabilities(df, model, ax=ax_probas)

    # Plot hists of probas
    fig_scores = plt.figure()
    ax_scores = fig_scores.add_subplot(1, 1, 1)

    plot_precision_recall(df, model, ax=ax_scores)

    if output is None:
        plt.show()
    else:
        with PdfPages(output) as pdf:
            for fig in (fig_roc, fig_scores, fig_probas):
                pdf.savefig(fig)
