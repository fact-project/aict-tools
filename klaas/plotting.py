import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import pandas as pd
from matplotlib.colors import LogNorm
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

    bin_edges = np.linspace(0, 1, model.n_estimators + 2)

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

    thresholds = np.linspace(0, 1, model.n_estimators + 2)
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
    ax.set_xlabel('Feature importances')
    ax.set_title('The {} most important features'.format(len(feature_names[:20])))
    ax.figure.tight_layout()
