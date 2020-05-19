import click
import logging
import matplotlib
import matplotlib.pyplot as plt
import joblib
import fact.io

from ..configuration import AICTConfig
from ..plotting import (
    plot_roc,
    plot_probabilities,
    plot_regressor_confusion,
    plot_bias_resolution,
    plot_feature_importances,
    plot_true_delta_delta
)

if matplotlib.get_backend() == 'pgf':
    from matplotlib.backends.backend_pgf import PdfPages
else:
    from matplotlib.backends.backend_pdf import PdfPages


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('performance_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('data_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('sign_model_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('disp_model_path', type=click.Path(exists=True, dir_okay=False))
@click.option('-o', '--output', type=click.Path(exists=False, dir_okay=False))
@click.option('-k', '--key', help='HDF5 key for hdf5 for performance_path', default='data')
@click.option('-k_data', '--key_data', help='HDF5 key for hdf5 for data_path', default='events')
def main(configuration_path, performance_path, data_path, sign_model_path, disp_model_path, output, key, key_data):
    ''' Create some performance evaluation plots for the separator '''
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger()

    log.info('Loading perfomance data')
    df = fact.io.read_data(performance_path, key=key)

    log.info('Loading original data')
    df_data = fact.io.read_data(data_path, key=key_data)

    log.info('Loading sign model')
    sign_model = joblib.load(sign_model_path)

    log.info('Loading disp model')
    disp_model = joblib.load(disp_model_path)

    model_config = AICTConfig.from_yaml(configuration_path).disp
    figures = []

    # Plot confusion
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    ax.set_title('Reconstructed vs. True |disp| (log color scale)')
    plot_regressor_confusion(df, log_xy=False, ax=ax, label_str='disp', label_prediction_str='disp_prediction')
    ax.set_xlabel(r'$|disp|_{\mathrm{MC}} \,\, / \,\, \mathrm{m}$')
    ax.set_ylabel(r'$|disp|_{\mathrm{Est}} \,\, / \,\, \mathrm{m}$')

    # Plot confusion
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    ax.set_title('Reconstructed vs. True |disp| (linear color scale)')
    plot_regressor_confusion(df, log_xy=False, log_z=False, ax=ax, label_str='disp', label_prediction_str='disp_prediction')
    ax.set_xlabel(r'$|disp|_{\mathrm{MC}} \,\, / \,\, \mathrm{m}$')
    ax.set_ylabel(r'$|disp|_{\mathrm{Est}} \,\, / \,\, \mathrm{m}$')

  #  # Plot bias/resolution
  #  figures.append(plt.figure())
  #  ax = figures[-1].add_subplot(1, 1, 1)
  #  ax.set_title('Bias and Resolution for |disp|')
  #  plot_bias_resolution(df, bins=15, log_x=False, ax=ax, label_str='disp', label_prediction_str='disp_prediction')
  #  ax.set_xlabel(r'$|disp|_{\mathrm{true}} \,\, / \,\, \mathrm{m}$')

    # Plot ROC
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    plot_roc(df, sign_model, ax=ax, label_str='sign', label_proba_str='sign_probabilities')

    # Plot hists of probas
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    plot_probabilities(df, sign_model, ax=ax, classnames={-1.0:r'$-$', 1.0:r'$+$'}, label_str='sign', label_proba_str='sign_score')

    # Plot feature importances sign
    if hasattr(sign_model, 'feature_importances_'):
        figures.append(plt.figure())
        ax = figures[-1].add_subplot(1, 1, 1)
        ax.set_title('Feature Importance sign')

        features = model_config.features

        plot_feature_importances(sign_model, features, ax=ax)

    # Plot feature importances disp
    if hasattr(disp_model, 'feature_importances_'):
        figures.append(plt.figure())
        ax = figures[-1].add_subplot(1, 1, 1)
        ax.set_title('Feature Importance absolute disp')


        features = model_config.features

        plot_feature_importances(disp_model, features, ax=ax)

    # Plot delta_true - delta
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    plot_true_delta_delta(df_data, ax)


    if output is None:
        plt.show()
    else:
        with PdfPages(output) as pdf:
            for fig in figures:
                fig.tight_layout(pad=0)
                pdf.savefig(fig)

if __name__ == '__main__':
    main()