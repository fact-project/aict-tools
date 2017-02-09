import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def plot_roc(performace_df, model, ax=None):

    ax = ax or plt.gca()

    ax.axvline(0, color='lightgray')
    ax.axvline(1, color='lightgray')
    ax.axhline(0, color='lightgray')
    ax.axhline(1, color='lightgray')

    roc_aucs = []
    fprs = []
    tprs = []

    thresholds = np.linspace(0, 1, model.n_estimators + 2)

    for it, df in performace_df.groupby('cv_fold'):

        fpr = []
        tpr = []
        for threshold in thresholds:
            (tp, fn), (fp, tn) = metrics.confusion_matrix(
                df['label'], (df['probabilities'] >= threshold).astype(int)
            )
            fpr.append(fp / (fp + tn))
            tpr.append(tp / (tp + fn))

        fprs.append(fpr)
        tprs.append(tpr)

        roc_aucs.append(metrics.roc_auc_score(df['label'], df['probabilities']))

        ax.plot(
            fpr, tpr,
            color='lightgray', lw=0.66 * plt.rcParams['lines.linewidth'],
            label='Single CV ROC Curve' if it == 0 else None
        )

    ax.set_title('Mean area under curve: {:.4f} ± {:.4f}'.format(
        np.mean(roc_aucs), np.std(roc_aucs)
    ))

    ax.plot(np.mean(fprs, axis=0), np.mean(tprs, axis=0), label='Mean ROC curve')
    ax.legend()
    ax.set_aspect(1)

    ax.set_xlabel('false positive rate')
    ax.set_ylabel('true positive rate')

    return ax


def plot_probabilities(performace_df, model, ax=None, classnames=('Proton', 'Gamma')):

    ax = ax or plt.gca()

    bin_edges = np.linspace(0, 1, model.n_estimators + 2)

    for label, df in performace_df.groupby('label'):
        ax.hist(
            df.probabilities,
            bins=bin_edges, label=classnames[label], histtype='step',
        )

    ax.legend()
    ax.set_xlabel('{} confidence'.format(classnames[1]))


def plot_precision_recall(performace_df, model, ax=None, beta=0.1):

    ax = ax or plt.gca()

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
