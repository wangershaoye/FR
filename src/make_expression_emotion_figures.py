"""Two emotion-based facial-expression figures (both 3 x 6, landscape).

Figure S1  expression_emotions_vs_ratings
    Rows = the three per-face mean ratings; columns = the six basic emotions.
    Each panel: emotion score (x) vs mean rating (y), with the Spearman rho.
    The top (artificiality) row is flat for every emotion -> orthogonality.

Figure S2  expression_emotion_robustness_grid
    Rows = the three rating pairs; columns = the six basic emotions.
    Each panel: per-participant baseline partial rho (x) vs the partial rho after
    adding that single emotion as a covariate (y); grey = participants, red
    diamond = Fisher-z pooled estimate, dashed = identity line.

Place in src/ (next to common.py / manuscript_figures.py).
Run from the repo root:  python src/make_expression_emotion_figures.py

Expected pooled rho for S2 (sanity check against console):
  Art x Trust|Att : anger -0.327 disgust -0.319 fear -0.327 happy -0.329 sad -0.327 surprise -0.326  (baseline -0.326)
  Art x Att|Trust : anger +0.028 disgust +0.021 fear +0.031 happy +0.027 sad +0.032 surprise +0.035  (baseline +0.037)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
import pingouin as pg

HERE = Path(__file__).resolve().parent
if (HERE / "common.py").exists():
    sys.path.insert(0, str(HERE))
elif (HERE / "src" / "common.py").exists():
    sys.path.insert(0, str(HERE / "src"))
from common import load_data, EXPRESSION_FEATURE_PATH, FIGURE_DIR
from manuscript_figures import set_house_style, save_figure, FIG_W, GREY, ACCENT, RHO, TIMES

EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
EMO_LABEL = {e: e.capitalize() for e in EMOTIONS}
ART, ATT, TR = "Artificiality", "Attractiveness", "Trustworthiness"
RATINGS = [("Mean perceived\nartificiality", ART),
           ("Mean\ntrustworthiness", TR),
           ("Mean\nattractiveness", ATT)]
PAIRS = [("Artificiality " + TIMES + " Trustworthiness", ART, TR, [ATT]),
         ("Artificiality " + TIMES + " Attractiveness", ART, ATT, [TR]),
         ("Attractiveness " + TIMES + " Trustworthiness", ATT, TR, [ART])]
PT, DIAG, MEAN = "#4a4a4a", "#b0b0b0", ACCENT


def figure_orthogonality(face_means, features):
    set_house_style()
    merged = features.merge(face_means.reset_index(), on="stimulus_id", how="inner")
    fig, axes = plt.subplots(3, 6, figsize=(FIG_W * 1.7, 7.6), sharex="col", sharey="row")
    for r, (ylabel, col) in enumerate(RATINGS):
        for c, emo in enumerate(EMOTIONS):
            ax = axes[r][c]
            xx = merged[emo].to_numpy()
            yy = merged[col].to_numpy()
            ax.scatter(xx, yy, s=7, color=GREY, alpha=0.35, edgecolors="none", zorder=1)
            coef = np.polyfit(xx, yy, 1)
            xs = np.linspace(xx.min(), xx.max(), 100)
            ax.plot(xs, np.polyval(coef, xs), "-", color="0.15", lw=1.6, zorder=3)
            rho, p = stats.spearmanr(xx, yy)
            ptxt = "n.s." if p >= .05 else ("p < .001" if p < .001 else "p = %.3f" % p)
            ax.set_title(RHO + " = %.3f (%s)" % (rho, ptxt), fontsize=13)
            if r == 2:
                ax.set_xlabel(EMO_LABEL[emo])
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=13)
    fig.tight_layout()
    return fig


def figure_robustness(tl):
    set_house_style()
    plt.rcParams.update({"axes.titlesize": 12, "axes.labelsize": 12,
                         "xtick.labelsize": 12, "ytick.labelsize": 12})

    def pp(x, y, covar):
        return np.array([pg.partial_corr(data=g, x=x, y=y, covar=covar, method="spearman")["r"].values[0]
                         for _, g in tl.groupby("participant_id")])

    def zmean(r):
        return np.tanh(np.arctanh(np.clip(r, -0.999999, 0.999999)).mean())

    base = {name: pp(x, y, cov) for (name, x, y, cov) in PAIRS}
    adj = {(name, emo): pp(x, y, cov + [emo])
           for (name, x, y, cov) in PAIRS for emo in EMOTIONS}

    fig, axes = plt.subplots(3, 6, figsize=(FIG_W * 1.7, 7.4))
    for r, (name, x, y, cov) in enumerate(PAIRS):
        allv = np.concatenate([base[name]] + [adj[(name, e)] for e in EMOTIONS])
        lo, hi = allv.min() - 0.04, allv.max() + 0.04
        for c, emo in enumerate(EMOTIONS):
            ax = axes[r][c]
            b, a = base[name], adj[(name, emo)]
            ax.plot([lo, hi], [lo, hi], "--", color=DIAG, lw=1.2, zorder=1)
            ax.scatter(b, a, s=16, c=PT, alpha=0.5, edgecolors="none", zorder=2)
            ax.scatter([zmean(b)], [zmean(a)], s=90, marker="D", facecolor=MEAN,
                       edgecolor="white", linewidth=1.0, zorder=4)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_box_aspect(1)
            if r == 0:
                ax.set_title(EMO_LABEL[emo])
            if r == 2:
                ax.set_xlabel("Baseline partial " + RHO)
            if c == 0:
                ax.set_ylabel(name + "\nadjusted partial " + RHO, fontsize=11)
            ax.text(0.05, 0.95, "%+.3f" % zmean(a), transform=ax.transAxes,
                    fontsize=8.5, va="top", ha="left", color=MEAN)
            print("%-34s %-9s base=%+.3f adj=%+.3f" % (name, emo, zmean(base[name]), zmean(a)), flush=True)
    legend = [Line2D([0], [0], marker="o", color="w", mfc=PT, alpha=0.6, markersize=8, label="Individual participant"),
              Line2D([0], [0], marker="D", color="w", mfc=MEAN, mec="white", markersize=10, label="Pooled estimate")]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.01))
    fig.align_ylabels(axes[:, 0])
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def main():
    ratings = load_data()
    features = pd.read_csv(EXPRESSION_FEATURE_PATH)[["stimulus_id", *EMOTIONS]]
    face_means = ratings.groupby("stimulus_id")[[ART, TR, ATT]].mean()
    tl = ratings.merge(features, on="stimulus_id", how="left")

    save_figure(figure_orthogonality(face_means, features),
                "expression_emotions_vs_ratings", FIGURE_DIR)
    save_figure(figure_robustness(tl),
                "expression_emotion_robustness_grid", FIGURE_DIR)
    print("Saved both figures to", FIGURE_DIR)


if __name__ == "__main__":
    main()
