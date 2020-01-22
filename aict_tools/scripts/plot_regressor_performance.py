import click
import logging
import matplotlib
import matplotlib.pyplot as plt
import joblib
import fact.io

from ..configuration import AICTConfig
from ..plotting import (
    plot_regressor_confusion,
    plot_bias_resolution,
    plot_feature_importances,
)

if matplotlib.get_backend() == 'pgf':
    from matplotlib.backends.backend_pgf import PdfPages
else:
    from matplotlib.backends.backend_pdf import PdfPages


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
    ax.set_title('Reconstructed vs. True Energy (log color scale)')
    plot_regressor_confusion(df, ax=ax)

    # Plot confusion
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    ax.set_title('Reconstructed vs. True Energy (linear color scale)')
    plot_regressor_confusion(df, log_z=False, ax=ax)

    # Plot bias/resolution
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    ax.set_title('Bias and Resolution')
    plot_bias_resolution(df, bins=15, ax=ax)

    if hasattr(model, 'feature_importances_'):
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
                fig.tight_layout(pad=0)
                pdf.savefig(fig)
