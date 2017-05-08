import click
import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.externals import joblib
import yaml
import pandas as pd

from ..plotting import (
    plot_roc,
    plot_probabilities,
    plot_precision_recall,
    plot_feature_importances,
    plot_binned_auc,
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
    df = pd.read_hdf(performance_path, key)

    log.info('Loading model')
    model = joblib.load(model_path)

    with open(configuration_path) as f:
        config = yaml.load(f)

    figures = []

    # Plot rocs
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    plot_roc(df, model, ax=ax)

    # Plot roc_auc vs. size
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)

    ax.set_title('Area under ROC curve vs. Size')
    plot_binned_auc(
        df,
        key='size',
        xlabel=r'$\log_{10}(\mathtt{Size})$',
        n_bins=15,
        ax=ax,
    )

    # Plot roc_auc vs. size
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)

    ax.set_title('Area under ROC curve vs. MC Energy')
    plot_binned_auc(
        df,
        key='energy',
        xlabel=r'$\log_{10}(E \,/\, \mathrm{GeV})$',
        n_bins=15,
        ax=ax,
    )

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

    training_variables = config['training_variables']
    if 'feature_generation' in config:
        training_variables.extend(config['feature_generation']['features'])

    plot_feature_importances(model, training_variables, ax=ax)

    if output is None:
        plt.show()
    else:
        with PdfPages(output) as pdf:
            for fig in figures:
                pdf.savefig(fig)
