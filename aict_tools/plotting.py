import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap
from sklearn.calibration import CalibratedClassifierCV

def hex2rgb(s):
    h = s.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2 ,4))

def make_colormap(colors, pos=None):
    if pos is None:
        pos = np.linspace(0.0, 1.0, len(colors))
    if type(colors[0]) == str:
        colors = [hex2rgb(c) for c in colors]
    assert len(pos) == len(colors), 'pos and colors must be same length'
    cdict = {
        'red':   [[p, c[0], c[0]] for p, c in zip(pos, colors)],
        'green': [[p, c[1], c[1]] for p, c in zip(pos, colors)],
        'blue':  [[p, c[2], c[2]] for p, c in zip(pos, colors)]
    }
    return LinearSegmentedColormap('CustomMap', cdict)

colors = ['#1F2648', '#283554', '#4f6e86', '#719bae', '#abced5', '#e6f5f9']
cmap = make_colormap(colors)


def plot_regressor_confusion(performance_df, log_xy=True, log_z=True, ax=None, 
                            label='label', pred='label_prediction'):

    ax = ax or plt.gca()

    label = performance_df[label].copy()
    prediction = performance_df[pred].copy()

    if log_xy is True:
        label = np.log10(label)
        prediction = np.log10(prediction)

    limits = [
        min(prediction.min(), label.min()),
        max(prediction.max(), label.max()),
    ]

    counts, x_edges, y_edges, img = ax.hist2d(
        label,
        prediction,
        bins=[100, 100],
        range=[limits, limits],
        cmap=cmap,
        norm=LogNorm() if log_z is True else None,
    )
    ax.set_aspect(1)
    ax.figure.colorbar(img, ax=ax)

    return ax


def plot_bias_resolution(performace_df, bins=10, ax=None, 
                        keys=('bias', 'resolution', 'resolution_quantiles')):
    df = performace_df.copy()

    ax = ax or plt.gca()

    if np.isscalar(bins):
        bins = np.logspace(
            np.log10(df.label.min()),
            np.log10(df.label.max()),
            bins + 1
        )

    df['bin'] = np.digitize(df.label, bins)
    df['rel_error'] = (df.label_prediction - df.label) / df.label

    binned = pd.DataFrame(index=np.arange(1, len(bins)))
    binned['center'] = 0.5 * (bins[:-1] + bins[1:])
    binned['width'] = np.diff(bins)

    grouped = df.groupby('bin')
    binned['bias'] = grouped['rel_error'].mean()
    binned['bias_median'] = grouped['rel_error'].median()
    binned['lower_sigma'] = grouped['rel_error'].agg(lambda s: np.percentile(s, 15))
    binned['upper_sigma'] = grouped['rel_error'].agg(lambda s: np.percentile(s, 85))
    binned['resolution_quantiles'] = (binned.upper_sigma - binned.lower_sigma) / 2
    binned['resolution'] = grouped['rel_error'].std()

    for key in keys:
        ax.errorbar(
            binned['center'],
            binned[key],
            xerr=0.5 * binned['width'],
            label=key,
            linestyle='',
        )
    ax.legend()
    ax.set_xscale('log')

    return ax

def plot_bias(performance_df, bins=15, ax=None, log_x=True, 
              label='label', pred='label_prediction'):
    df = performance_df.copy()

    ax = ax or plt.gca()

    if log_x:
        if np.isscalar(bins):
            bins = np.logspace(
                np.log10(df[label].min()),
                np.log10(df[label].max()),
                bins + 1
            )
    else:
        if np.isscalar(bins):
            bins = np.linspace(
                df[label].min(),
                df[label].max(),
                bins + 1
            )

    df['bin'] = np.digitize(df[label], bins)
    df['rel_error'] = (df[pred] - df[label]) / df[label]

    binned = pd.DataFrame(index=np.arange(1, len(bins)))
    binned['center'] = 0.5 * (bins[:-1] + bins[1:])
    binned['width'] = np.diff(bins)

    grouped = df.groupby('bin')
    binned['Mean'] = grouped['rel_error'].mean()
    binned['Median'] = grouped['rel_error'].median()

    for key in ('Mean', 'Median'):

        ax.errorbar(
            binned['center'][1:],
            binned[key][1:],
            xerr=0.5 * binned['width'][1:],
            label=key,
            linestyle='', lw=2,
        )
    ax.legend()
    if log_x:
        ax.set_xscale('log')
    ax.axhline(0, color='lightgray', zorder=0)

    return ax

def plot_resolution(performace_df, bins=15, log_x=True, ax=None,
                    label='label', pred='label_prediction'):
    df = performace_df.copy()

    ax = ax or plt.gca()

    if log_x:
        if np.isscalar(bins):
            bins = np.logspace(
                np.log10(df[label].min()),
                np.log10(df[label].max()),
                bins + 1
            )
    else:
        if np.isscalar(bins):
            bins = np.linspace(
                df[label].min(),
                df[label].max(),
                bins + 1
            )

    df['bin'] = np.digitize(df[label], bins)
    df['rel_error'] = (df[pred] - df[label]) / df[label]

    binned = pd.DataFrame(index=np.arange(1, len(bins)))
    binned['center'] = 0.5 * (bins[:-1] + bins[1:])
    binned['width'] = np.diff(bins)

    grouped = df.groupby('bin')
    binned['lower_sigma'] = grouped['rel_error'].agg(lambda s: np.percentile(s, 15))
    binned['upper_sigma'] = grouped['rel_error'].agg(lambda s: np.percentile(s, 85))
    binned['Interquantile Range'] = (binned.upper_sigma - binned.lower_sigma) / 2
    binned['Standard Deviation'] = grouped['rel_error'].std()

    for key in ('Standard Deviation', 'Interquantile Range'):
        ax.errorbar(
            binned['center'][1:],
            binned[key][1:],
            xerr=0.5 * binned['width'][1:],
            label=key,
            linestyle='', lw=2,
        )
    ax.legend()
    if log_x:
        ax.set_xscale('log')

    return ax


def plot_roc(performance_df, model, ax=None, label='label', pred='probabilities'):
    ax = ax or plt.gca()

    ax.axvline(0, color='lightgray')
    ax.axvline(1, color='lightgray')
    ax.axhline(0, color='lightgray')
    ax.axhline(1, color='lightgray')

    color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]

    roc_aucs = []

    mean_fpr, mean_tpr, _ = metrics.roc_curve(performance_df[label], 
                                              performance_df[pred])
    for it, df in performance_df.groupby('cv_fold'):
        fpr, tpr, _ = metrics.roc_curve(df[label], df[pred])

        roc_aucs.append(metrics.roc_auc_score(df[label], df[pred]))

        ax.plot(
            fpr, tpr,
            color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0], 
            alpha=0.5, zorder=0,
            lw=0.66 * plt.rcParams['lines.linewidth'],
            label='Single CV ROC Curve' if it == 0 else None
        )

    ax.text(0.9, 0.1,
            'AUC: {:.4f} ± {:.4f}'.format(np.mean(roc_aucs), np.std(roc_aucs)),
            transform=ax.transAxes,
            horizontalalignment='right')

    ax.plot(mean_fpr, mean_tpr, label='Mean ROC curve')
    ax.set_aspect(1)

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.figure.tight_layout()

    return ax


def plot_probabilities(performance_df, model, ax=None, 
                        classnames=('Proton', 'Gamma'), 
                        label='label', pred='probabilities'):

    ax = ax or plt.gca()

    if isinstance(model, CalibratedClassifierCV):
        model = model.base_estimator

    bin_edges = np.linspace(0, 1, model.n_estimators + 2)
    bin_mids = (bin_edges[1:] + bin_edges[:-1]) * 0.5

    y_max = 0

    for lbl, df in performance_df.groupby(label):
        hist, _ = np.histogram(df[pred], bins=bin_edges)
        ax.errorbar(bin_mids, hist, xerr=np.diff(bin_edges) * 0.5, 
            linestyle='', lw=3,
            label=classnames[lbl] if classnames[0] == 'Proton' else None)
        if y_max < np.max(hist):
            y_max = np.max(hist)

    ax.legend(loc='upper center')
    ax.set_xlabel('{} Score'.format(classnames[1]))
    ax.set_yscale('log')
    ax.set_ylim([1, 10**(np.log10(y_max)+1)])
    ax.figure.tight_layout()

    return ax


def plot_precision_recall(performance_df, model, ax=None, beta=0.1, classname='Gamma'):

    ax = ax or plt.gca()

    if isinstance(model, CalibratedClassifierCV):
        model = model.base_estimator


    ax.axvline(0, color='lightgray')
    ax.axvline(1, color='lightgray')
    ax.axhline(0, color='lightgray')
    ax.axhline(1, color='lightgray')

    thresholds = np.linspace(0, 1, model.n_estimators + 2)

    for it, df in performance_df.groupby('cv_fold'):
        precision = []
        recall = []
        f_beta = []

        for threshold in thresholds:

            prediction = (df['probabilities'] >= threshold).astype('int')
            label = df['label']

            precision.append(metrics.precision_score(label, prediction))
            recall.append(metrics.recall_score(label, prediction))
            f_beta.append(metrics.fbeta_score(label, prediction, beta=beta))

        ax.plot(thresholds, precision, zorder=0, alpha=0.5)
        ax.plot(thresholds, recall, zorder=0, alpha=0.5)
        ax.plot(thresholds, f_beta, zorder=0, alpha=0.5)

    precision = []
    recall = []
    f_beta = []

    for threshold in thresholds:

        prediction = (performance_df.probabilities.values >= threshold).astype('int')
        label = performance_df.label.values

        precision.append(metrics.precision_score(label, prediction))
        recall.append(metrics.recall_score(label, prediction))
        f_beta.append(metrics.fbeta_score(label, prediction, beta=beta))

    ax.plot(thresholds, precision, label='Purity', zorder=3)
    ax.plot(thresholds, recall, label='Efficiency', zorder=2)
    ax.plot(thresholds, f_beta, label='$f_{{{:.2f}}}$'.format(beta), zorder=1)

    ax.legend(loc=(0.1,0.1))
    ax.set_xlabel(f'Threshold of {classname} Score')
    ax.set_ylabel('Cross Validation Score')
    ax.figure.tight_layout()

    return ax


def plot_feature_importances(model, feature_names, ax=None):

    ax = ax or plt.gca()

    y_pos = np.arange(len(feature_names[:20]))

    if isinstance(model, CalibratedClassifierCV):
        model = model.base_estimator

    if hasattr(model, 'estimators_'):

        df = pd.DataFrame(index=feature_names)

        feature_importances = [est.feature_importances_ for est in model.estimators_]

        df['mean'] = np.mean(feature_importances, axis=0)
        df['p_low'] = np.percentile(feature_importances, 15.87, axis=0)
        df['p_high'] = np.percentile(feature_importances, 84.13, axis=0)

        df.sort_values('mean', inplace=True)
        df = df.tail(20)

        ax.barh(
            y_pos,
            df['mean'].values,
            xerr=[df['mean'] - df['p_low'], df['p_high'] - df['mean']],
            alpha=0.7
        )

    else:
        df = pd.DataFrame(index=feature_names)
        df['mean'] = model.feature_importances_

        df.sort_values('mean', inplace=True)
        df = df.tail(20)

        ax.barh(
            y_pos,
            df['mean'].values
        )

    ax.set_ylim(-0.5, y_pos.max() + 0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df.index.values)
    ax.set_xlabel('Feature Importance')
    ax.figure.tight_layout()

    return ax
