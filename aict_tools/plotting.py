import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import warnings

from sklearn import metrics
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.calibration import CalibratedClassifierCV

from .preprocessing import delta_error, horizontal_to_camera


def plot_regressor_confusion(
    performance_df,
    log_xy=True,
    log_z=True,
    ax=None,
    label_column="label",
    prediction_column="label_prediction",
    energy_unit="GeV",
):

    ax = ax or plt.gca()

    label = performance_df[label_column].copy()

    prediction = performance_df[prediction_column].copy()

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
        ax.set_xlabel(
            rf"$\log_{{10}}(E_{{\mathrm{{MC}}}} \,\, / \,\, \mathrm{{{energy_unit}}})$"
        )
        ax.set_ylabel(
            rf"$\log_{{10}}(E_{{\mathrm{{Est}}}} \,\, / \,\, \mathrm{{{energy_unit}}})$"
        )
    else:
        ax.set_xlabel(rf"$E_{{\mathrm{{MC}}}} \,\, / \,\, \mathrm{{{energy_unit}}}$")
        ax.set_ylabel(rf"$E_{{\mathrm{{Est}}}} \,\, / \,\, \mathrm{{{energy_unit}}}$")

    return ax


def plot_bias_resolution(
    performance_df,
    bins=10,
    ax=None,
    label_column="label",
    prediction_column="label_prediction",
    energy_unit="GeV",
):
    df = performance_df.copy()

    ax = ax or plt.gca()

    if np.isscalar(bins):
        bins = np.logspace(
            np.log10(df[label_column].min()), np.log10(df[label_column].max()), bins + 1
        )

    df["bin"] = np.digitize(df[label_column], bins)
    df["rel_error"] = (df[prediction_column] - df[label_column]) / df[label_column]

    binned = pd.DataFrame(index=np.arange(1, len(bins)))
    binned["center"] = 0.5 * (bins[:-1] + bins[1:])
    binned["width"] = np.diff(bins)

    grouped = df.groupby("bin")
    binned["bias"] = grouped["rel_error"].mean()
    binned["bias_median"] = grouped["rel_error"].median()
    binned["lower_sigma"] = grouped["rel_error"].agg(lambda s: np.percentile(s, 15))
    binned["upper_sigma"] = grouped["rel_error"].agg(lambda s: np.percentile(s, 85))
    binned["resolution_quantiles"] = (binned.upper_sigma - binned.lower_sigma) / 2
    binned["resolution"] = grouped["rel_error"].std()
    binned = binned[grouped.size() > 100]  # at least fifty events

    for key in ("bias", "resolution", "resolution_quantiles"):
        if matplotlib.get_backend() == "pgf" or plt.rcParams["text.usetex"]:
            label = key.replace("_", r"\_")
        else:
            label = key

        ax.errorbar(
            binned["center"],
            binned[key],
            xerr=0.5 * binned["width"],
            label=label,
            linestyle="",
        )
    ax.legend()
    ax.set_xscale("log")
    ax.set_xlabel(
        rf"$\log_{{10}}(E_{{\mathrm{{MC}}}} \,\, / \,\, \mathrm{{{energy_unit}}})$"
    )

    return ax


def plot_roc(
    performance_df,
    model,
    ax=None,
    label_column="label",
    score_column="scores",
):

    ax = ax or plt.gca()

    ax.axvline(0, color="lightgray")
    ax.axvline(1, color="lightgray")
    ax.axhline(0, color="lightgray")
    ax.axhline(1, color="lightgray")

    roc_aucs = []

    mean_fpr, mean_tpr, _ = metrics.roc_curve(
        performance_df[label_column],
        performance_df[score_column],
    )
    for it, df in performance_df.groupby("cv_fold"):

        fpr, tpr, _ = metrics.roc_curve(df[label_column], df[score_column])

        roc_aucs.append(metrics.roc_auc_score(df[label_column], df[score_column]))

        ax.plot(
            fpr,
            tpr,
            color="lightgray",
            lw=0.66 * plt.rcParams["lines.linewidth"],
            label="Single CV ROC Curve" if it == 0 else None,
        )

    ax.set_title(
        "Mean area under curve: {:.4f} ± {:.4f}".format(
            np.mean(roc_aucs), np.std(roc_aucs)
        )
    )

    ax.plot(mean_fpr, mean_tpr, label="Mean ROC curve")
    ax.legend()
    ax.set_aspect(1)

    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.figure.tight_layout()

    return ax


def plot_scores(
    performance_df,
    model,
    ax=None,
    xlabel="score",
    classnames={0: "Proton", 1: "Gamma"},
    label_column="label",
    score_column="score",
):

    ax = ax or plt.gca()

    if isinstance(model, CalibratedClassifierCV):
        model = model.base_estimator

    n_bins = (model.n_estimators + 1) if hasattr(model, "n_estimators") else 100
    bin_edges = np.linspace(
        performance_df[score_column].min(),
        performance_df[score_column].max(),
        n_bins + 1,
    )
    for label, df in performance_df.groupby(label_column):
        ax.hist(
            df[score_column],
            bins=bin_edges,
            label=classnames[label],
            histtype="step",
        )

    ax.set_xlabel(xlabel)
    ax.legend()
    ax.figure.tight_layout()


def plot_precision_recall(
    performance_df, model, score_column="score", ax=None, beta=0.1
):

    ax = ax or plt.gca()

    if isinstance(model, CalibratedClassifierCV):
        model = model.base_estimator

    n_bins = (model.n_estimators + 1) if hasattr(model, "n_estimators") else 100
    thresholds = np.linspace(0, 1, n_bins + 1)
    precision = []
    recall = []
    f_beta = []

    ax.axvline(0, color="lightgray")
    ax.axvline(1, color="lightgray")
    ax.axhline(0, color="lightgray")
    ax.axhline(1, color="lightgray")

    for threshold in thresholds:

        prediction = (performance_df[score_column] >= threshold).astype("int")
        label = performance_df.label.values

        precision.append(metrics.precision_score(label, prediction))
        recall.append(metrics.recall_score(label, prediction))
        f_beta.append(metrics.fbeta_score(label, prediction, beta=beta))

    ax.plot(thresholds, precision, label="precision")
    ax.plot(thresholds, recall, label="recall")
    ax.plot(thresholds, f_beta, label="$f_{{{:.2f}}}$".format(beta))

    ax.legend()
    ax.set_xlabel("prediction threshold")
    ax.figure.tight_layout()


def plot_feature_importances(model, feature_names, ax=None, max_features=20):

    ax = ax or plt.gca()

    ypos = np.arange(1, len(feature_names[:max_features]) + 1)

    if plt.rcParams["text.usetex"] or matplotlib.get_backend() == "pgf":
        feature_names = [f.replace("_", r"\_") for f in feature_names]
    feature_names = np.array(feature_names)

    if isinstance(model, CalibratedClassifierCV):
        model = model.base_estimator

    if hasattr(model, "estimators_"):
        feature_importances = np.array(
            [est.feature_importances_ for est in np.array(model.estimators_).ravel()]
        )

        idx = np.argsort(np.median(feature_importances, axis=0))[-max_features:]

        ax.boxplot(
            feature_importances[:, idx],
            vert=False,
            sym="",  # no outliers
            medianprops={"color": "C0"},
        )

        y_jittered = np.random.normal(ypos, 0.1, size=feature_importances[:, idx].shape)

        for imp, y in zip(feature_importances.T[idx], y_jittered.T):
            res = ax.scatter(imp, y, color="C1", alpha=0.5, lw=0, s=5)
            res.set_rasterized(True)

    else:
        feature_importances = model.feature_importances_
        idx = np.argsort(feature_importances)[-max_features:]

        ax.barh(ypos, feature_importances[idx])

    ax.set_ylim(ypos[0] - 0.5, ypos[-1] + 0.5)
    ax.set_yticks(ypos)
    ax.set_yticklabels(feature_names[idx])
    ax.set_xlabel("Feature importance")
    if len(feature_names) > max_features:
        ax.set_title("The {} most important features".format(max_features))
    ax.figure.tight_layout()


def plot_true_delta_delta(data_df, config, ax=None):

    df = data_df.copy()
    source_x, source_y = horizontal_to_camera(df, config)
    true_delta = np.arctan2(
        source_y - df[config.cog_y_column],
        source_x - df[config.cog_x_column],
    )

    ax.hist(true_delta - df[config.delta_column], bins=100, histtype="step")
    ax.figure.tight_layout()
    ax.set_xlabel(r"$\delta_{true}\,-\,\delta$")
    return ax


def plot_energy_dependent_disp_metrics(
    df, true_energy_column, energy_unit="GeV", fig=None
):

    df = df.copy()
    edges = 10 ** np.arange(
        np.log10(df[true_energy_column].min()),
        np.log10(df[true_energy_column].max()),
        0.2,
    )
    df["bin_idx"] = np.digitize(df[true_energy_column], edges)

    def accuracy(group):
        return metrics.accuracy_score(
            group.sign,
            group.sign_prediction,
        )

    def r2(group):
        return metrics.r2_score(
            np.abs(group.disp),
            group.disp_prediction,
        )

    # discard under and overflow
    df = df[(df["bin_idx"] != 0) & (df["bin_idx"] != len(edges))]

    binned = pd.DataFrame(
        {
            "e_center": 0.5 * (edges[1:] + edges[:-1]),
            "e_low": edges[:-1],
            "e_high": edges[1:],
            "e_width": np.diff(edges),
        },
        index=pd.Series(np.arange(1, len(edges)), name="bin_idx"),
    )

    r2_scores = pd.DataFrame(index=binned.index)
    accuracies = pd.DataFrame(index=binned.index)
    counts = pd.DataFrame(index=binned.index)

    with warnings.catch_warnings():
        # warns when there are less than 2 events for calculating metrics,
        # but we throw those away anyways
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        for cv_fold, cv in df.groupby("cv_fold"):
            grouped = cv.groupby("bin_idx")
            accuracies[cv_fold] = grouped.apply(accuracy)
            r2_scores[cv_fold] = grouped.apply(r2)
            counts[cv_fold] = grouped.size()

    binned["r2_score"] = r2_scores.mean(axis=1)
    binned["r2_std"] = r2_scores.std(axis=1)
    binned["accuracy"] = accuracies.mean(axis=1)
    binned["accuracy_std"] = accuracies.std(axis=1)
    # at least 10 events in each crossval iteration
    binned["valid"] = (counts > 100).any(axis=1)
    binned = binned.query("valid")

    fig = fig or plt.figure()

    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

    ax1.errorbar(
        binned.e_center,
        binned.accuracy,
        yerr=binned.accuracy_std,
        xerr=binned.e_width / 2,
        ls="",
    )
    ax1.set_ylabel(r"Accuracy for $\mathrm{sgn} \mathtt{disp}$")

    ax2.errorbar(
        binned.e_center,
        binned.r2_score,
        yerr=binned.r2_std,
        xerr=binned.e_width / 2,
        ls="",
    )
    ax2.set_ylabel(r"$r^2$ score for $|\mathtt{disp}|$")
    ax2.set_ylim(None, 1)

    ax2.set_xlabel(r"$E_{\mathrm{true}} \,\,/\,\," + rf" \mathrm{{{energy_unit}}}$")
    ax2.set_xscale("log")

    return fig


def plot_energy_dependent_dxdy_metrics(df, true_energy_column, energy_unit='GeV', fig=None):

    df = df.copy()
    edges = 10**np.arange(
        np.log10(df[true_energy_column].min()),
        np.log10(df[true_energy_column].max()),
        0.2
    )
    df['bin_idx'] = np.digitize(df[true_energy_column], edges)

    def r2_dx(group):
        return metrics.r2_score(
            group.dx,
            group.dx_prediction,
        )

    def r2_dy(group):
        return metrics.r2_score(
            group.dy,
            group.dy_prediction,
        )

    # discard under and overflow
    df = df[(df['bin_idx'] != 0) & (df['bin_idx'] != len(edges))]

    binned = pd.DataFrame({
        'e_center': 0.5 * (edges[1:] + edges[:-1]),
        'e_low': edges[:-1],
        'e_high': edges[1:],
        'e_width': np.diff(edges),
    }, index=pd.Series(np.arange(1, len(edges)), name='bin_idx'))

    r2_dx_scores = pd.DataFrame(index=binned.index)
    r2_dy_scores = pd.DataFrame(index=binned.index)
    counts = pd.DataFrame(index=binned.index)

    with warnings.catch_warnings():
        # warns when there are less than 2 events for calculating metrics,
        # but we throw those away anyways
        warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
        for cv_fold, cv in df.groupby('cv_fold'):
            grouped = cv.groupby('bin_idx')
            r2_dx_scores[cv_fold] = grouped.apply(r2_dx)
            r2_dy_scores[cv_fold] = grouped.apply(r2_dy)
            counts[cv_fold] = grouped.size()

    binned['r2_dx_score'] = r2_dx_scores.mean(axis=1)
    binned['r2_dx_std'] = r2_dx_scores.std(axis=1)
    binned['r2_dy_score'] = r2_dy_scores.mean(axis=1)
    binned['r2_dy_std'] = r2_dy_scores.std(axis=1)

    # at least 10 events in each crossval iteration
    binned['valid'] = (counts > 100).any(axis=1)
    binned = binned.query('valid')

    fig = fig or plt.figure()

    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

    ax1.errorbar(
        binned.e_center, binned.r2_dx_score,
        yerr=binned.r2_dx_std, xerr=binned.e_width / 2,
        ls='',
    )
    ax1.set_ylabel(r'$r^2$ score for $dx$')
    ax1.set_ylim(None, 1)
    ax1.grid()

    ax2.errorbar(
        binned.e_center, binned.r2_dy_score,
        yerr=binned.r2_dy_std, xerr=binned.e_width / 2,
        ls='',
    )
    ax2.set_ylabel(r'$r^2$ score for $dy$')
    ax2.set_ylim(None, 1)
    ax2.grid()

    ax2.set_xlabel(
        r'$E_{\mathrm{wahr}} \,\,/\,\,' + rf' \mathrm{{{energy_unit}}}$'
    )
    ax2.set_xscale('log')

    return fig


def plot_rocauc_vs_size(
    df,
    size_column="size",
    label_column="label",
    score_column="scores",
    ax=None,
):

    df = df.copy()
    edges = np.geomspace(
        df[size_column].min(),
        df[size_column].max(),
        50,
    )
    df["bin_idx"] = np.digitize(df[size_column], edges)

    def roc_auc_score(group):
        try:
            return metrics.roc_auc_score(group[label_column], group[score_column])
        except:
            return np.nan

    # discard under and overflow
    df = df[(df["bin_idx"] != 0) & (df["bin_idx"] != len(edges))]

    binned = pd.DataFrame(
        {
            "size_center": 0.5 * (edges[1:] + edges[:-1]),
            "size_low": edges[:-1],
            "size_high": edges[1:],
            "size_width": np.diff(edges),
        },
        index=pd.Series(np.arange(1, len(edges)), name="bin_idx"),
    )

    roc_aucs = pd.DataFrame(index=binned.index)
    counts = pd.DataFrame(index=binned.index)

    with warnings.catch_warnings():
        # warns when there are less than 2 events for calculating metrics,
        # but we throw those away anyways
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        for cv_fold, cv in df.groupby("cv_fold"):
            grouped = cv.groupby("bin_idx")
            roc_aucs[cv_fold] = grouped.apply(roc_auc_score)
            counts[cv_fold] = grouped.size()

    binned["roc_auc"] = roc_aucs.mean(axis=1)
    binned["roc_auc_std"] = roc_aucs.std(axis=1)
    binned["valid"] = (counts > 50).any(axis=1)
    binned = binned.query("valid")

    ax = ax or plt.gca()

    ax.errorbar(
        binned.size_center,
        binned.roc_auc,
        yerr=binned.roc_auc_std,
        xerr=binned.size_width / 2,
        ls="",
    )
    ax.set_ylabel(r"$A_{\mathrm{ROC}}$")
    ax.set_xlabel(size_column)
    ax.set_xscale("log")

    return ax
