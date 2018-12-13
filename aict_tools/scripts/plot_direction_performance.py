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
    plot_regressor_confusion,
    plot_bias_resolution,
    plot_bias,
    plot_resolution,
    plot_feature_importances,
)


@click.command()
@click.argument('configuration_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('performance_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('model_path_sign', type=click.Path(exists=True, dir_okay=False))
@click.argument('model_path_disp', type=click.Path(exists=True, dir_okay=False))
@click.option('-o', '--output', type=click.Path(exists=False, dir_okay=False))
@click.option('-k', '--key', help='HDF5 key for hdf5', default='data')
@click.option('-p', '--parameter', type=click.Choice(['energy', 'disp']), 
                default='energy', help='Parameter to be estimated')
def main(configuration_path, performance_path, model_path_sign, model_path_disp, 
        output, key, parameter):
    ''' Create some performance evaluation plots for the separator '''
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger()

    log.info('Loading perfomance data')
    df = fact.io.read_data(performance_path, key=key)

    log.info('Loading model sign')
    model_sign = joblib.load(model_path_sign)

    log.info('Loading model disp')
    model_disp = joblib.load(model_path_disp)

    config = AICTConfig.from_yaml(configuration_path)
    model_config = config.disp
    figures = []

    # Plot ROC
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    plot_roc(df, model_sign, ax=ax, label='sign', pred='sign_probabilities')

    # Plot hists of probas
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    ax = plot_probabilities(df, model_sign, ax=ax, 
       label='sign', pred='sign_probabilities', 
       classnames=('Minus', 'Plus'))

    # Plot confusion
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    ax = plot_regressor_confusion(df, ax=ax, log_xy=False,
                            label='disp', pred='disp_prediction')
    ax.plot([0,500], [0,500], color='#D03A3B', alpha=0.5)
    ax.set_xlabel(r'$disp_{\mathrm{true}} \,\, / \,\, \mathrm{mm}$')
    ax.set_ylabel(r'$disp_{\mathrm{rec}} \,\, / \,\, \mathrm{mm}$')


    # Plot confusion for different energies
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(2, 2, 1)
    ax = plot_regressor_confusion(df[df[config.energy.target_column]<1], 
        ax=ax, log_xy=False, label='disp', pred='disp_prediction')
    ax.set_ylabel(r'$disp_{\mathrm{rec}} \,\, / \,\, \mathrm{mm}$')
    ax.set_xlim([0,400])
    ax.set_ylim([0,400])
    ax.plot([0,500], [0,500], color='#D03A3B', alpha=0.5)
    ax.text(0.1,0.9,'< 1 TeV', fontsize=8,
        transform=ax.transAxes, horizontalalignment='left')

    ax = figures[-1].add_subplot(2, 2, 2)
    ax = plot_regressor_confusion(df[(df[config.energy.target_column]>1) 
        & (df[config.energy.target_column]<10)], 
        ax=ax, log_xy=False, label='disp', pred='disp_prediction')
    ax.set_xlim([0,400])
    ax.set_ylim([0,400])
    ax.plot([0,500], [0,500], color='#D03A3B', alpha=0.5)
    ax.text(0.1,0.9,'1 - 10 TeV', fontsize=8,
        transform=ax.transAxes, horizontalalignment='left')

    ax = figures[-1].add_subplot(2, 2, 3)
    ax = plot_regressor_confusion(df[(df[config.energy.target_column]>10) 
        & (df[config.energy.target_column]<100)], 
        ax=ax, log_xy=False, label='disp', pred='disp_prediction')
    ax.set_xlabel(r'$disp_{\mathrm{true}} \,\, / \,\, \mathrm{mm}$')
    ax.set_ylabel(r'$disp_{\mathrm{rec}} \,\, / \,\, \mathrm{mm}$')
    ax.set_xlim([0,400])
    ax.set_ylim([0,400])
    ax.plot([0,500], [0,500], color='#D03A3B', alpha=0.5)
    ax.text(0.1,0.9,'10 - 100 TeV', fontsize=8,
        transform=ax.transAxes, horizontalalignment='left')

    ax = figures[-1].add_subplot(2, 2, 4)
    ax = plot_regressor_confusion(df[df[config.energy.target_column]>100], 
        ax=ax, log_xy=False, label='disp', pred='disp_prediction')
    ax.set_xlabel(r'$disp_{\mathrm{true}} \,\, / \,\, \mathrm{mm}$')
    ax.set_xlim([0,400])
    ax.set_ylim([0,400])
    ax.plot([0,500], [0,500], color='#D03A3B', alpha=0.5)
    ax.text(0.1,0.9,'> 100 TeV', fontsize=8,
        transform=ax.transAxes, horizontalalignment='left')


    # Plot bias
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    ax = plot_bias(df, bins=15, ax=ax, log_x=False,
        label='disp', pred='disp_prediction')
    ax.set_xlabel(r'$disp_{\mathrm{true}} \,\, / \,\, \mathrm{mm}$')
    ax.set_ylabel('Bias')

    # Plot resolution
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    ax = plot_resolution(df, bins=15, ax=ax, log_x=False,
                    label='disp', pred='disp_prediction')
    ax.set_xlabel(r'$disp_{\mathrm{true}} \,\, / \,\, \mathrm{mm}$')
    ax.set_ylabel('Resolution')

    # Plot feature importances
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    features = model_config.features
    ax = plot_feature_importances(model_disp, features, ax=ax)
    ax.text(0.95, 0.05, 'Disp Regression',
            transform=ax.transAxes, horizontalalignment='right')

    # Plot feature importances
    figures.append(plt.figure())
    ax = figures[-1].add_subplot(1, 1, 1)
    features = model_config.features
    ax = plot_feature_importances(model_sign, features, ax=ax)
    ax.text(0.95, 0.05, 'Sign Classification',
            transform=ax.transAxes, horizontalalignment='right')


    if output is None:
        plt.show()
    else:
        with PdfPages(output) as pdf:
            for fig in figures:
                pdf.savefig(fig)
