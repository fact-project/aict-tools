import matplotlib.pyplot as plt
import numpy as np
import click
from klaas import read_data
plt.style.use('ggplot')


@click.command()
@click.argument('predictions_path', type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True))
@click.argument('output_path', type=click.Path(exists=False, dir_okay=False, file_okay=True))
@click.option('--n_bins', '-b', default=102,  help='Number of bins to plot for performance plots')
def plot_prediction_histogram(predictions_path, output_path, n_bins):
    print("Loading data ")
    predictions = read_data(predictions_path)
    # embed()
    fig, ax = plt.subplots(1)
    # embed()
    proton_probabilities = predictions.query('label == 0').probabilities
    gamma_probabilities = predictions.query('label == 1').probabilities

    bins = np.linspace(0, 1, n_bins)

    ax.hist(gamma_probabilities, bins=bins, normed=True,
            color='#cc4368', alpha=0.6, label='Signal')
    ax.hist(proton_probabilities, bins=bins, normed=True,
            color='#3c84d7', alpha=0.6, label='Background')
    ax.set_xlabel('prediction threshold')
    ax.legend(bbox_to_anchor=(0.0, 1.03, 1, .2), loc='lower center',
              ncol=2, borderaxespad=0., fancybox=True, framealpha=0.0)

    plt.savefig(output_path)


if __name__ == '__main__':
    plot_prediction_histogram()
