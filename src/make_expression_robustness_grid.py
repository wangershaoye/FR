"""Grid version of the expression-robustness figure.

Baseline vs adjusted participant-wise partial Spearman rho, for several
facial-expression covariate sets. Rows = adjustment (AU12 / AU6 / happiness /
6 basic emotions); columns = the three rating pairs. Grey dots = participants,
red diamond = Fisher-z pooled estimate, dashed = identity line. Reuses the
house style, paths and save routine of the existing pipeline.

Place this file in src/ (next to common.py and manuscript_figures.py).
Run from the repo root:   python src/make_expression_robustness_grid.py
Output: outputs/figures/expression_partial_correlation_robustness_grid.{png,pdf}

Expected pooled rho (sanity check against console output):
  Art x Trust | Attr :  base -0.326  AU12 -0.326  AU6 -0.323  happy -0.329  6emo -0.317
  Art x Attr | Trust :  base +0.037  AU12 +0.033  AU6 +0.036  happy +0.027  6emo +0.016
  Att x Trust | Art  :  base +0.439  AU12 +0.421  AU6 +0.425  happy +0.414  6emo +0.423
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pingouin as pg

HERE = Path(__file__).resolve().parent
if (HERE / "common.py").exists():
    sys.path.insert(0, str(HERE))
elif (HERE / "src" / "common.py").exists():
    sys.path.insert(0, str(HERE / "src"))
from common import load_data, EXPRESSION_FEATURE_PATH, FIGURE_DIR
from manuscript_figures import set_house_style, save_figure, FIG_W, ACCENT, RHO, TIMES

ART, ATT, TR = "Artificiality", "Attractiveness", "Trustworthiness"
PT, DIAG, MEAN = "#4a4a4a", "#b0b0b0", ACCENT

# rows: (row label, extra covariate columns added on top of the 3rd rating)
# Trim this list to shrink the figure (e.g. keep only AU12 + 6 basic emotions).
ADJUSTMENTS = [
    ("AU12 (smile)", ["AU12"]),
    ("AU6 (cheek raiser)", ["AU6"]),
    ("Happiness", ["happiness"]),
    ("6 basic emotions", ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]),
]
# columns: (title, x, y, baseline covariate list)
PAIRS = [
    ("Artificiality " + TIMES + " Trustworthiness", ART, TR, [ATT]),
    ("Artificiality " + TIMES + " Attractiveness", ART, ATT, [TR]),
    ("Attractiveness " + TIMES + " Trustworthiness", ATT, TR, [ART]),
]


def main():
    set_house_style()
    ratings = load_data()
    cols = sorted({c for _, cs in ADJUSTMENTS for c in cs})
    features = pd.read_csv(EXPRESSION_FEATURE_PATH)[["stimulus_id", *cols]]
    tl = ratings.merge(features, on="stimulus_id", how="left")

    def pp(x, y, covar):
        return np.array([
            pg.partial_corr(data=g, x=x, y=y, covar=covar, method="spearman")["r"].values[0]
            for _, g in tl.groupby("participant_id")
        ])

    def zmean(r):
        return np.tanh(np.arctanh(np.clip(r, -0.999999, 0.999999)).mean())

    base_by_pair = {name: pp(x, y, cov) for (name, x, y, cov) in PAIRS}

    nrows = len(ADJUSTMENTS)
    fig, axes = plt.subplots(nrows, 3, figsize=(FIG_W * 1.5, 3.7 * nrows), squeeze=False)
    for r, (adj_name, adj_cols) in enumerate(ADJUSTMENTS):
        for c, (name, x, y, cov_base) in enumerate(PAIRS):
            ax = axes[r][c]
            base = base_by_pair[name]
            adj = pp(x, y, cov_base + adj_cols)
            lo = min(base.min(), adj.min()) - 0.04
            hi = max(base.max(), adj.max()) + 0.04
            ax.plot([lo, hi], [lo, hi], "--", color=DIAG, lw=1.3, zorder=1)
            ax.scatter(base, adj, s=24, c=PT, alpha=0.55, edgecolors="none", zorder=2)
            ax.scatter([zmean(base)], [zmean(adj)], s=120, marker="D", facecolor=MEAN,
                       edgecolor="white", linewidth=1.1, zorder=4)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_box_aspect(1)
            if r == 0:
                ax.set_title(name)
            ax.set_xlabel("Baseline partial " + RHO if r == nrows - 1 else "")
            ax.set_ylabel(adj_name + "\nadjusted partial " + RHO if c == 0 else "")
            ax.text(0.05, 0.96,
                    "pooled " + RHO + "\nbaseline  %+.3f\nadjusted  %+.3f" % (zmean(base), zmean(adj)),
                    transform=ax.transAxes, fontsize=9, va="top", ha="left", linespacing=1.4)
            print("row=%-18s %-34s base=%+.3f adj=%+.3f"
                  % (adj_name, name, zmean(base), zmean(adj)), flush=True)

    legend = [
        Line2D([0], [0], marker="o", color="w", mfc=PT, alpha=0.6, markersize=8, label="Individual participant"),
        Line2D([0], [0], marker="D", color="w", mfc=MEAN, mec="white", markersize=10, label="Pooled estimate"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save_figure(fig, "expression_partial_correlation_robustness_grid", FIGURE_DIR)
    print("Saved to", FIGURE_DIR)


if __name__ == "__main__":
    main()
