import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV


def plot_regressor_confusion(performace_df, log_xy=True, log_z=True, ax=None):

    ax = ax or plt.gca()

    label = performace_df.label.copy()
    prediction = performace_df.label_prediction.copy()

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
        norm=LogNorm() if log_z is True else None,
    )
    img.set_rasterized(True)
    ax.set_aspect(1)
    ax.figure.colorbar(img, ax=ax)

    if log_xy is True:
        ax.set_xlabel(r'$\log_{10}(E_{\mathrm{MC}} \,\, / \,\, \mathrm{GeV})$')
        ax.set_ylabel(r'$\log_{10}(E_{\mathrm{Est}} \,\, / \,\, \mathrm{GeV})$')
    else:
        ax.set_xlabel(r'$E_{\mathrm{MC}} \,\, / \,\, \mathrm{GeV}$')
        ax.set_ylabel(r'$E_{\mathrm{Est}} \,\, / \,\, \mathrm{GeV}$')

    return ax


def plot_bias_resolution(performace_df, bins=10, ax=None):
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

    for key in ('bias', 'resolution', 'resolution_quantiles'):
        if matplotlib.get_backend() == 'pgf' or plt.rcParams['text.usetex']:
            label = key.replace('_', r'\_')
        else:
            label = key

        ax.errorbar(
            binned['center'],
            binned[key],
            xerr=0.5 * binned['width'],
            label=label,
            linestyle='',
        )
    ax.legend()
    ax.set_xscale('log')

    return ax


def plot_roc(performace_df, model, ax=None):

    ax = ax or plt.gca()

    ax.axvline(0, color='lightgray')
    ax.axvline(1, color='lightgray')
    ax.axhline(0, color='lightgray')
    ax.axhline(1, color='lightgray')

    roc_aucs = []

    mean_fpr, mean_tpr, _ = metrics.roc_curve(performace_df['label'], performace_df['probabilities'])
    for it, df in performace_df.groupby('cv_fold'):

        fpr, tpr, _ = metrics.roc_curve(df['label'], df['probabilities'])

        roc_aucs.append(metrics.roc_auc_score(df['label'], df['probabilities']))

        ax.plot(
            fpr, tpr,
            color='lightgray', lw=0.66 * plt.rcParams['lines.linewidth'],
            label='Single CV ROC Curve' if it == 0 else None
        )

    ax.set_title('Mean area under curve: {:.4f} ± {:.4f}'.format(
        np.mean(roc_aucs), np.std(roc_aucs)
    ))

    ax.plot(mean_fpr, mean_tpr, label='Mean ROC curve')
    ax.legend()
    ax.set_aspect(1)

    ax.set_xlabel('false positive rate')
    ax.set_ylabel('true positive rate')
    ax.figure.tight_layout()

    return ax


def plot_probabilities(performace_df, model, ax=None, classnames=('Proton', 'Gamma')):

    ax = ax or plt.gca()

    if isinstance(model, CalibratedClassifierCV):
        model = model.base_estimator

    n_bins = (model.n_estimators + 1) if hasattr(model, 'n_estimators') else 100
    bin_edges = np.linspace(0, 1, n_bins + 1)

    for label, df in performace_df.groupby('label'):
        ax.hist(
            df.probabilities,
            bins=bin_edges, label=classnames[label], histtype='step',
        )

    ax.legend()
    ax.set_xlabel('{} confidence'.format(classnames[1]))
    ax.figure.tight_layout()


def plot_precision_recall(performace_df, model, ax=None, beta=0.1):

    ax = ax or plt.gca()

    if isinstance(model, CalibratedClassifierCV):
        model = model.base_estimator

    n_bins = (model.n_estimators + 1) if hasattr(model, 'n_estimators') else 100
    thresholds = np.linspace(0, 1, n_bins + 1)
    precision = []
    recall = []
    f_beta = []

    ax.axvline(0, color='lightgray')
    ax.axvline(1, color='lightgray')
    ax.axhline(0, color='lightgray')
    ax.axhline(1, color='lightgray')
    for threshold in thresholds:

        prediction = (performace_df.probabilities.values >= threshold).astype('int')
        label = performace_df.label.values

        precision.append(metrics.precision_score(label, prediction))
        recall.append(metrics.recall_score(label, prediction))
        f_beta.append(metrics.fbeta_score(label, prediction, beta=beta))

    ax.plot(thresholds, precision, label='precision')
    ax.plot(thresholds, recall, label='recall')
    ax.plot(thresholds, f_beta, label='$f_{{{:.2f}}}$'.format(beta))

    ax.legend()
    ax.set_xlabel('prediction threshold')
    ax.figure.tight_layout()


def plot_feature_importances(model, feature_names, ax=None, max_features=20):

    ax = ax or plt.gca()

    ypos = np.arange(1, len(feature_names[:max_features]) + 1)

    if plt.rcParams['text.usetex'] or matplotlib.get_backend() == 'pgf':
        feature_names = [f.replace('_', r'\_') for f in feature_names]
    feature_names = np.array(feature_names)

    if isinstance(model, CalibratedClassifierCV):
        model = model.base_estimator

    if hasattr(model, 'estimators_'):
        feature_importances = np.array([
            est.feature_importances_
            for est in np.array(model.estimators_).ravel()
        ])

        idx = np.argsort(np.median(feature_importances, axis=0))[-max_features:]

        ax.boxplot(
            feature_importances[:, idx],
            vert=False,
            sym='',  # no outliers
            medianprops={'color': 'C0'}
        )

        y_jittered = np.random.normal(ypos, 0.1, size=feature_importances[:, idx].shape)

        for imp, y in zip(feature_importances.T[idx], y_jittered.T):
            res = ax.scatter(imp, y, color='C1', alpha=0.5, lw=0, s=5)
            res.set_rasterized(True)

    else:
        feature_importances = model.feature_importances_
        idx = np.argsort(feature_importances)[-max_features:]

        ax.barh(
            ypos,
            feature_importances[idx]
        )

    ax.set_ylim(ypos[0] - 0.5, ypos[-1] + 0.5)
    ax.set_yticks(ypos)
    ax.set_yticklabels(feature_names[idx])
    ax.set_xlabel('Feature importance')
    if len(feature_names) > max_features:
        ax.set_title('The {} most important features'.format(max_features))
    ax.figure.tight_layout()
