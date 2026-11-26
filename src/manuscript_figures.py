"""Single source of truth for the manuscript's non-image figures.

Both the analysis pipeline (``src/*.py`` -> ``outputs/figures/``) and
``notebooks/figures.ipynb`` import these functions, so the released figures and
the inline notebook figures are guaranteed to be identical and to follow the
manuscript's visualization scheme (the shared house style used throughout the
revised figures). Each function recomputes from the public release data and
returns a Matplotlib :class:`~matplotlib.figure.Figure`; callers either save it
(the pipeline, via :func:`save_figure`) or display it (the notebook).

Figures 2 and R1 are example-face montages that require the stimulus images and
are rendered only in the notebook; they are intentionally not part of this
module, which is image-free like the rest of the pipeline.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats


# --- shared house style (from the project's revised figure scripts) ---
FIG_W = 7.48
FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND = 13, 12, 10, 10.5
LW_MAIN, LW_REF = 2.5, 1.0
GREY, ACCENT, REF = "0.65", "#C0392B", "0.20"

DIMENSIONS = ["Artificiality", "Trustworthiness", "Attractiveness"]

RHO = "ρ"       # Greek small rho
NDASH = "–"     # en dash
TIMES = "×"     # multiplication sign
PM = "±"        # plus-minus sign


def set_house_style():
    """Apply the shared manuscript house style to Matplotlib rcParams."""
    mpl.rcParams.update({
        "font.size": FS_TICK, "axes.titlesize": FS_TITLE, "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "legend.fontsize": FS_LEGEND,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 1.0,
        "xtick.major.width": 1.0, "ytick.major.width": 1.0,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    try:
        mpl.rcParams["font.family"] = "Arial"
    except Exception:
        pass


def save_figure(fig, stem, figure_dir, close=True):
    """Save *fig* as ``<stem>.png`` (300 dpi) and ``<stem>.pdf`` in *figure_dir*."""
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(figure_dir / f"{stem}.pdf", bbox_inches="tight")
    if close:
        plt.close(fig)


def _fisher_mean_ci(r, alpha=0.05):
    """Fisher-z pooled mean correlation and its (1 - alpha) t confidence interval."""
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    n = len(r)
    z = np.arctanh(r)
    mz, se = z.mean(), z.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(1 - alpha / 2, n - 1)
    return np.tanh(mz), np.tanh(mz - tcrit * se), np.tanh(mz + tcrit * se)


def figure_zero_order(corr):
    """Figure 3 - participant-wise zero-order Spearman rho distributions.

    Grey histograms of the three per-participant correlations, each with the
    Fisher-z group-mean rho (red line), its 95% CI (red band), and rho = 0.
    """
    set_house_style()
    pairs = [("rho_art_att", "Artificiality " + NDASH + "\nAttractiveness"),
             ("rho_art_tru", "Artificiality " + NDASH + "\nTrustworthiness"),
             ("rho_att_tru", "Attractiveness " + NDASH + "\nTrustworthiness")]
    stats_tbl = {col: _fisher_mean_ci(corr[col]) for col, _ in pairs}
    all_vals = np.concatenate([corr[c].dropna().values for c, _ in pairs])
    xmax = np.ceil(np.nanmax(np.abs(all_vals)) * 10) / 10.0
    xlim = (-xmax, xmax)
    bins = np.linspace(xlim[0], xlim[1], 25)

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, 3.4), sharey=True)
    for ax, (col, name) in zip(axes, pairs):
        ax.hist(corr[col].dropna().values, bins=bins, color=GREY, edgecolor="black",
                linewidth=0.5, zorder=1)
        m, lo, hi = stats_tbl[col]
        ax.axvspan(lo, hi, color=ACCENT, alpha=0.20, zorder=2)
        ax.axvline(m, color=ACCENT, linewidth=LW_MAIN, zorder=4)
        ax.axvline(0, color=REF, linewidth=LW_REF, zorder=3)
        ax.set_title(name)
        ax.set_xlabel("Spearman's " + RHO)
        ax.set_xlim(xlim)
    axes[0].set_ylabel("Number of participants")
    handles = [Line2D([0], [0], color=ACCENT, lw=LW_MAIN, label="Mean " + RHO),
               Patch(facecolor=ACCENT, alpha=0.20, label="95% CI")]
    fig.tight_layout(rect=[0, 0.11, 1, 1])
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=2, frameon=False)
    return fig


def figure_partial_joint(partial):
    """Figure 4 - joint distribution of participant-wise partial correlations.

    Each grey point is one participant: x = partial rho(artificiality,
    attractiveness | trustworthiness); y = partial rho(artificiality,
    trustworthiness | attractiveness). Marginal histograms, dotted zero lines,
    and the dashed identity line (y = x).
    """
    set_house_style()
    x = partial["p_art_att_given_tru"].values
    y = partial["p_art_tru_given_att"].values
    FS_L4, FS_T4 = 16, 13
    mabs = float(np.nanmax(np.abs(np.concatenate([x, y]))))
    L = max(0.7, np.ceil(mabs * 10) / 10.0)
    LIM = (-L, L)
    bin_edges = np.linspace(-L, L, 21)
    cnt = np.concatenate([np.histogram(x, bins=bin_edges)[0],
                          np.histogram(y, bins=bin_edges)[0]])
    cmax = int(np.ceil(cnt.max() / 10.0) * 10)

    fig = plt.figure(figsize=(9.5, 8.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                          wspace=0.03, hspace=0.03)
    ax_main = fig.add_subplot(gs[1, 0])
    ax_main.scatter(x, y, color="grey", edgecolor="black", alpha=0.7, s=42,
                    linewidth=0.5, zorder=2)
    ax_main.axhline(0, color="grey", linestyle="dotted", zorder=1)
    ax_main.axvline(0, color="grey", linestyle="dotted", zorder=1)
    ax_main.plot([-1, 1], [-1, 1], color="black", linestyle="dashed", linewidth=1, zorder=1)
    ax_main.set_xlim(LIM)
    ax_main.set_ylim(LIM)
    ax_main.set_xlabel("Partial Correlation between Artificiality and Attractiveness",
                       fontsize=FS_L4, labelpad=12)
    ax_main.set_ylabel("Partial Correlation between Artificiality and Trustworthiness",
                       fontsize=FS_L4)
    ax_main.tick_params(labelsize=FS_T4)
    for sp in ax_main.spines.values():
        sp.set_visible(True)
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_top.hist(x, bins=bin_edges, color="grey", edgecolor="black")
    ax_top.set_ylabel("Number of participants", fontsize=FS_T4)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.tick_params(labelsize=FS_T4)
    ax_top.set_xlim(LIM)
    ax_top.set_yticks(range(0, cmax + 1, 10))
    ax_top.set_ylim(0, cmax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_right.hist(y, bins=bin_edges, orientation="horizontal", color="grey", edgecolor="black")
    ax_right.set_xlabel("Number of\nparticipants", fontsize=FS_T4)
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.tick_params(labelsize=FS_T4)
    ax_right.set_ylim(LIM)
    ax_right.set_xticks(range(0, cmax + 1, 10))
    ax_right.set_xlim(0, cmax)
    return fig


def figure_gee_predictions(trials):
    """Figure 5 - population-averaged GEE predictions with observed means.

    Marginal predictions of attractiveness (A) and trustworthiness (B) across the
    artificiality scale, each holding the other rating at its sample mean, with
    95% bands and the observed per-level mean ratings (+/- 95% CI) overlaid. Both
    panels share a common y-axis. Explicit font sizes keep the rendered and saved
    figures identical regardless of the caller's rcParams.
    """
    set_house_style()
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    gee = trials.copy()
    gee["pid"] = pd.factorize(gee["participant_id"])[0]
    ART, ATT, TR = "Artificiality", "Attractiveness", "Trustworthiness"
    mean_tr = gee[TR].mean()
    mean_at = gee[ATT].mean()

    def fit(formula):
        return smf.gee(formula, "pid", data=gee, cov_struct=sm.cov_struct.Exchangeable(),
                       family=sm.families.Gaussian()).fit()

    m_at = fit("Attractiveness ~ Artificiality + Trustworthiness")
    m_tr = fit("Trustworthiness ~ Artificiality + Attractiveness")
    grid = np.linspace(1, 5, 100)

    def band(model, cov, cm):
        names = ["Intercept", "Artificiality", cov]
        b = np.array([model.params[n] for n in names])
        V = model.cov_params().loc[names, names].values
        P = np.array([np.array([1., a, cm]) @ b for a in grid])
        S = np.array([np.sqrt(np.array([1., a, cm]) @ V @ np.array([1., a, cm])) for a in grid])
        return P, P - 1.96 * S, P + 1.96 * S

    at_p, at_lo, at_hi = band(m_at, "Trustworthiness", mean_tr)
    tr_p, tr_lo, tr_hi = band(m_tr, "Attractiveness", mean_at)

    def obs(y):
        g = gee.groupby(ART)[y]
        m = g.mean()
        se = g.std(ddof=1) / np.sqrt(g.count())
        return m.index.values.astype(float), m.values, 1.96 * se.values

    ox, oat, eat = obs(ATT)
    _, otr, etr = obs(TR)

    RED, OBS = "#c34335", "#2b2b2b"
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.95))
    YL, YT = (2.1, 3.55), [2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4]
    for ax, (lab, name, P, LO, HI, om, oe, leg) in zip(axes, [
            ("A", "Attractiveness", at_p, at_lo, at_hi, oat, eat, False),
            ("B", "Trustworthiness", tr_p, tr_lo, tr_hi, otr, etr, True)]):
        ax.fill_between(grid, LO, HI, color=RED, alpha=0.20, lw=0, zorder=2, label="95% CI")
        ax.plot(grid, P, "-", color=RED, lw=1.8, zorder=3, label="Adjusted GEE prediction")
        ax.errorbar(ox, om, yerr=oe, fmt="o", mfc="white", mec=OBS, color=OBS, ecolor=OBS,
                    mew=1.0, ms=4.0, capsize=0, elinewidth=0.9, zorder=4,
                    label="Observed mean rating")
        ax.set_title(lab, loc="left", fontweight="bold", fontsize=11)
        ax.set_xlabel("Artificiality", fontsize=9)
        ax.set_ylabel(name, fontsize=9)
        ax.tick_params(labelsize=9)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlim(0.7, 5.3)
        ax.set_ylim(*YL)
        ax.set_yticks(YT)
        for sp in ax.spines.values():
            sp.set_linewidth(0.8)
        if leg:
            h, l = ax.get_legend_handles_labels()
            order = [1, 0, 2]
            ax.legend([h[i] for i in order], [l[i] for i in order], loc="upper right",
                      fontsize=7.5, frameon=False, handlelength=1.6)
    fig.tight_layout()
    return fig


def figure_nonlinearity(face_means):
    """Non-linearity (uncanny-valley) trends in the house style.

    Stimulus-level mean trustworthiness and attractiveness against mean perceived
    artificiality, with linear and quadratic fits and quintile means (+/- SE).
    *face_means* is a per-stimulus mean table with the three rating columns.
    """
    set_house_style()
    art = face_means["Artificiality"].to_numpy()

    def fits(x, y):
        xc = x - x.mean()
        c1 = np.polyfit(xc, y, 1)
        c2 = np.polyfit(xc, y, 2)
        xs = np.linspace(x.min(), x.max(), 200)
        xsc = xs - x.mean()
        return xs, np.polyval(c1, xsc), np.polyval(c2, xsc)

    quintile = pd.qcut(pd.Series(art), 5, labels=False, duplicates="drop").to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 3.6), sharex=True, sharey=True)
    for ax, (ylabel, col) in zip(axes, [("Mean trustworthiness", "Trustworthiness"),
                                        ("Mean attractiveness", "Attractiveness")]):
        yv = face_means[col].to_numpy()
        ax.scatter(art, yv, s=12, color=GREY, alpha=0.45, edgecolors="none", zorder=1)
        xs, lin, quad = fits(art, yv)
        ax.plot(xs, lin, "--", color=REF, lw=1.3, zorder=3)
        ax.plot(xs, quad, "-", color=ACCENT, lw=LW_MAIN, zorder=4)
        bx, by, be = [], [], []
        for level in np.unique(quintile[~pd.isna(quintile)]):
            m = quintile == level
            bx.append(art[m].mean())
            by.append(yv[m].mean())
            be.append(yv[m].std(ddof=1) / np.sqrt(m.sum()))
        ax.errorbar(bx, by, yerr=be, fmt="o", color="black", mfc="black", mec="black",
                    ms=5.5, capsize=2.5, elinewidth=1.2, ecolor="black", zorder=5)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Mean perceived artificiality (per face)")
    handles = [Line2D([0], [0], ls="--", color=REF, lw=1.3, label="Linear fit"),
               Line2D([0], [0], ls="-", color=ACCENT, lw=LW_MAIN, label="Quadratic fit"),
               Line2D([0], [0], marker="o", color="black", lw=0, label="Quintile mean " + PM + " SE")]
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=3, frameon=False)
    return fig


def figure_au12_vs_ratings(face_means, features):
    """Facial expression (AU12) versus the three per-face mean ratings.

    *face_means* is a per-stimulus mean table (indexed by ``stimulus_id``);
    *features* carries ``stimulus_id`` and the raw ``AU12`` smile intensity.
    """
    set_house_style()
    merged = features.merge(face_means.reset_index(), on="stimulus_id", how="inner")
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W * 1.55, 4.0))
    panels = [("Mean perceived artificiality", "Artificiality"),
              ("Mean trustworthiness", "Trustworthiness"),
              ("Mean attractiveness", "Attractiveness")]
    for ax, (ylabel, col) in zip(axes, panels):
        xx = merged["AU12"].to_numpy()
        yy = merged[col].to_numpy()
        ax.scatter(xx, yy, s=11, color=GREY, alpha=0.35, edgecolors="none", zorder=1)
        coef = np.polyfit(xx, yy, 1)
        xs = np.linspace(xx.min(), xx.max(), 100)
        ax.plot(xs, np.polyval(coef, xs), "-", color="0.15", lw=2.2, zorder=3)
        rho, p = stats.spearmanr(xx, yy)
        ptxt = "n.s." if p >= .05 else ("p < .001" if p < .001 else "p = %.3f" % p)
        ax.set_title("Spearman's " + RHO + " = %.3f (%s)" % (rho, ptxt))
        ax.set_xlabel("AU12 (smile intensity)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.03, 1.03)
    fig.tight_layout()
    return fig


def figure_expression_robustness(trials, au12):
    """Per-participant partial correlations before and after adjusting for AU12.

    For each rating pair, every participant's partial rho before (x) and after (y)
    adding AU12 as a covariate, with the identity line and the Fisher-z pooled
    estimate (red diamond). *trials* is trial-level data; *au12* carries
    ``stimulus_id`` and raw ``AU12``.
    """
    set_house_style()
    import pingouin as pg

    tl = trials.merge(au12, on="stimulus_id", how="left")
    ART, ATT, TR = "Artificiality", "Attractiveness", "Trustworthiness"

    def pp(x, y, covar):
        return np.array([pg.partial_corr(data=g, x=x, y=y, covar=covar, method="spearman")["r"].values[0]
                         for _, g in tl.groupby("participant_id")])

    def zmean(r):
        return np.tanh(np.arctanh(np.clip(r, -0.999999, 0.999999)).mean())

    pairs = [("Artificiality " + TIMES + " Trustworthiness", ART, TR, [ATT], [ATT, "AU12"]),
             ("Artificiality " + TIMES + " Attractiveness", ART, ATT, [TR], [TR, "AU12"]),
             ("Attractiveness " + TIMES + " Trustworthiness", ATT, TR, [ART], [ART, "AU12"])]
    PT, DIAG, MEAN = "#4a4a4a", "#b0b0b0", ACCENT
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W * 1.5, 4.5))
    for i, (ax, (name, x, y, cov_base, cov_adj)) in enumerate(zip(axes, pairs)):
        base = pp(x, y, cov_base)
        adj = pp(x, y, cov_adj)
        lo = min(base.min(), adj.min()) - 0.04
        hi = max(base.max(), adj.max()) + 0.04
        ax.plot([lo, hi], [lo, hi], "--", color=DIAG, lw=1.3, zorder=1)
        ax.scatter(base, adj, s=24, c=PT, alpha=0.55, edgecolors="none", zorder=2)
        ax.scatter([zmean(base)], [zmean(adj)], s=120, marker="D", facecolor=MEAN,
                   edgecolor="white", linewidth=1.1, zorder=4)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_box_aspect(1)
        ax.set_title(name)
        ax.set_xlabel("Baseline partial " + RHO)
        ax.set_ylabel("AU12-adjusted partial " + RHO if i == 0 else "")
        ax.text(0.05, 0.96, "pooled " + RHO + "\nbaseline  %+.3f\nadjusted  %+.3f" % (zmean(base), zmean(adj)),
                transform=ax.transAxes, fontsize=10, va="top", ha="left", linespacing=1.45)
    legend = [Line2D([0], [0], marker="o", color="w", mfc=PT, alpha=0.6, markersize=8,
                     label="Individual participant"),
              Line2D([0], [0], marker="D", color="w", mfc=MEAN, mec="white", markersize=10,
                     label="Pooled estimate")]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig
