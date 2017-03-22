import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from sklearn import metrics
from klaas import read_data
plt.style.use('ggplot')


@click.command()
@click.argument('predictions_path', type=click.Path(exists=True, dir_okay=False,))
@click.argument('output_path', type=click.Path(exists=False, dir_okay=False,))
@click.option('--inset/--no-inset', default=False)
def plot_roc_curves(predictions_path, output_path, inset):
    # print("Loading data ")
    predictions = read_data(predictions_path)
    fig, ax = plt.subplots(1)

    if inset:
        axins = zoomed_inset_axes(ax, 2.5, loc=1)

    aucs = []
    for cv, cv_df in predictions.groupby('cv_fold'):
        fpr, tpr, thresholds = metrics.roc_curve(
            cv_df.label, cv_df.probabilities
        )
        aucs.append(metrics.roc_auc_score(cv_df.label, cv_df.probabilities))
        ax.plot(fpr, tpr, linestyle='-', color='k', alpha=0.3)
        if inset:
            axins.plot(fpr, tpr, linestyle='-', color='k', alpha=0.3)

    ax.set_xlabel('False Positiv Rate')
    ax.set_ylabel('True Positiv Rate')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)

    if inset:
        axins.set_xlim(0, 0.15)
        axins.set_ylim(0.8, 1.0)
        axins.set_xticks([0.0, 0.05, 0.1, 0.15])
        axins.set_yticks([0.8, 0.85, 0.9, 0.95, 1.0])
        mark_inset(ax, axins, loc1=2, loc2=3, fc='none', ec='0.8')

    ax.text(0.95, 0.1, 'Area Under Curve: ${:.2f} \pm {:.4f}$'.format(np.array(aucs).mean(), np.array(aucs).std()),
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color='#404040', fontsize=11)

    # print('Saving plot')
    plt.savefig(output_path)


if __name__ == '__main__':
    plot_roc_curves()
