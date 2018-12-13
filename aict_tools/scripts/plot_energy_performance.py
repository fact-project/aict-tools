import click
import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.externals import joblib
from ..configuration import AICTConfig
import fact.io

from ..plotting import (
    plot_regressor_confusion,
    plot_bias_resolution,
    plot_bias,
    plot_resolution,
    plot_feature_importances,
)


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('performance_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('model_path', type=click.Path(exists=True, dir_okay=False))
@click.option('-o', '--output', type=click.Path(exists=False, dir_okay=False))
@click.option('-k', '--key', help='HDF5 key for hdf5', default='data')
def main(configuration_path, performance_path, model_path, output, key):
    ''' Create some performance evaluation plots for the separator '''
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger()

    log.info('Loading perfomance data')
    df = fact.io.read_data(performance_path, key=key)

    log.info('Loading model')
    model = joblib.load(model_path)

    model_config = AICTConfig.from_yaml(configuration_path).energy
    figures = []

    # Plot confusion
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    ax = plot_regressor_confusion(df, ax=ax)
    ax.set_xlabel(r'$\log_{10}(E_{\mathrm{true}} \,\, / \,\, \mathrm{TeV})$')
    ax.set_ylabel(r'$\log_{10}(E_{\mathrm{rec}} \,\, / \,\, \mathrm{TeV})$')

    # Plot bias
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    ax = plot_bias(df, bins=15, ax=ax)
    ax.set_xlabel(r'$E_{\mathrm{true}} \,\, / \,\, \mathrm{TeV}$')
    ax.set_ylabel('Bias')

    # Plot resolution
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    ax = plot_resolution(df, bins=15, ax=ax)
    ax.set_xlabel(r'$E_{\mathrm{true}} \,\, / \,\, \mathrm{TeV}$')
    ax.set_ylabel('Resolution')

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
