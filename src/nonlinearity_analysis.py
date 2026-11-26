"""Non-linearity (uncanny-valley) test for the artificiality-rating relationships.

Reviewer-requested robustness analysis. The question is whether perceived
artificiality relates to trustworthiness and attractiveness non-linearly, as a
U-shaped uncanny-valley dip would predict, or monotonically. Three complementary
views are reported:

1. Participant-wise quadratic regression. For each participant the rating is
   regressed on mean-centred artificiality and its square, and the linear and
   quadratic coefficients are tested against zero across participants with a
   one-sample t test (the same per-participant logic used for the correlations).
2. Stimulus-level aggregate regression. Per-stimulus mean ratings are regressed
   on mean-centred stimulus-mean artificiality; the added quadratic term is
   assessed with a nested F test and its change in R-squared, and the fitted
   vertex is checked against the observed artificiality range.
3. Ordered-bin trends and an extreme-group contrast. Stimuli are grouped into
   deciles and quintiles of mean artificiality to show the shape of the trend
   and to contrast the least- and most-artificial quintiles directly.

The script uses only numpy, pandas, scipy, and matplotlib and reads the public
trial-level ratings; no image models or expression features are required.
"""

import numpy as np
import pandas as pd
from scipy import stats

from common import FIGURE_DIR, TABLE_DIR, ensure_output_dirs, load_data, validate_data
import manuscript_figures as mf


RATINGS = [("Trustworthiness", "trustworthiness"), ("Attractiveness", "attractiveness")]


def participantwise_quadratic(df):
    """Per-participant linear and quadratic coefficients, each tested vs zero."""
    rows = []
    for rating, label in RATINGS:
        linear, quadratic = [], []
        for _, group in df.groupby("participant_id", sort=True):
            x = group["Artificiality"].to_numpy(dtype=float)
            y = group[rating].to_numpy(dtype=float)
            if np.unique(x).size < 3:
                continue
            x = x - x.mean()
            quad, lin, _ = np.polyfit(x, y, 2)
            linear.append(lin)
            quadratic.append(quad)
        linear = np.asarray(linear)
        quadratic = np.asarray(quadratic)
        t_lin = stats.ttest_1samp(linear, 0.0)
        t_quad = stats.ttest_1samp(quadratic, 0.0)
        rows.append(
            {
                "rating": label,
                "n_participants": linear.size,
                "mean_linear_coef": float(linear.mean()),
                "linear_t": float(t_lin.statistic),
                "linear_p": float(t_lin.pvalue),
                "mean_quadratic_coef": float(quadratic.mean()),
                "quadratic_t": float(t_quad.statistic),
                "quadratic_p": float(t_quad.pvalue),
            }
        )
    return pd.DataFrame(rows)


def _nested_quadratic(x_centered, y):
    """Linear vs quadratic fit statistics for one stimulus-level relationship."""
    n = len(y)
    sst = float(((y - y.mean()) ** 2).sum())

    c_lin, cov_lin = np.polyfit(x_centered, y, 1, cov=True)
    rss_lin = float(((y - np.polyval(c_lin, x_centered)) ** 2).sum())
    se_slope = float(np.sqrt(cov_lin[0, 0]))
    t_slope = c_lin[0] / se_slope
    p_slope = float(2.0 * stats.t.sf(abs(t_slope), n - 2))

    c_quad, cov_quad = np.polyfit(x_centered, y, 2, cov=True)
    rss_quad = float(((y - np.polyval(c_quad, x_centered)) ** 2).sum())
    se_quad = float(np.sqrt(cov_quad[0, 0]))
    t_quad = c_quad[0] / se_quad
    p_quad = float(2.0 * stats.t.sf(abs(t_quad), n - 3))

    f_stat = ((rss_lin - rss_quad) / 1.0) / (rss_quad / (n - 3))
    p_f = float(stats.f.sf(f_stat, 1, n - 3))
    vertex = float(-c_quad[1] / (2.0 * c_quad[0]))

    return {
        "linear_slope": float(c_lin[0]),
        "linear_p": p_slope,
        "R2_linear": 1.0 - rss_lin / sst,
        "quadratic_coef": float(c_quad[0]),
        "quadratic_t": float(t_quad),
        "quadratic_p": p_quad,
        "R2_quadratic": 1.0 - rss_quad / sst,
        "delta_R2": (rss_lin - rss_quad) / sst,
        "added_quadratic_F": float(f_stat),
        "added_quadratic_p": p_f,
        "vertex_centered": vertex,
        "artificiality_centered_min": float(x_centered.min()),
        "artificiality_centered_max": float(x_centered.max()),
        "vertex_in_range": bool(x_centered.min() <= vertex <= x_centered.max()),
    }


def stimulus_level(df):
    """Per-stimulus quadratic fits and least- vs most-artificial quintile contrast."""
    means = df.groupby("stimulus_id")[
        ["Artificiality", "Attractiveness", "Trustworthiness"]
    ].mean()
    art = means["Artificiality"].to_numpy()
    art_c = art - art.mean()
    quintile = pd.qcut(means["Artificiality"], 5, labels=False, duplicates="drop").to_numpy()
    low = quintile == np.nanmin(quintile)
    high = quintile == np.nanmax(quintile)

    rows = []
    for rating, label in RATINGS:
        y = means[rating].to_numpy()
        stats_row = {"rating": label, "n_stimuli": len(y)}
        stats_row["pearson_r"] = float(np.corrcoef(art, y)[0, 1])
        stats_row.update(_nested_quadratic(art_c, y))
        contrast = stats.ttest_ind(y[low], y[high])
        stats_row.update(
            {
                "q1_low_art_mean": float(y[low].mean()),
                "q5_high_art_mean": float(y[high].mean()),
                "low_minus_high": float(y[low].mean() - y[high].mean()),
                "extreme_t": float(contrast.statistic),
                "extreme_p": float(contrast.pvalue),
            }
        )
        rows.append(stats_row)
    return means, pd.DataFrame(rows)


def decile_table(means):
    """Decile means of each rating ordered by stimulus-mean artificiality."""
    decile = pd.qcut(means["Artificiality"], 10, labels=False, duplicates="drop")
    table = (
        means.assign(decile=decile)
        .groupby("decile")
        .agg(
            mean_artificiality=("Artificiality", "mean"),
            mean_trustworthiness=("Trustworthiness", "mean"),
            mean_attractiveness=("Attractiveness", "mean"),
            n_stimuli=("Artificiality", "size"),
        )
        .reset_index()
    )
    table["decile"] = table["decile"].astype(int) + 1
    return table


def run():
    ensure_output_dirs()
    df = load_data()
    validate_data(df)

    participant_summary = participantwise_quadratic(df)
    means, stimulus_summary = stimulus_level(df)
    deciles = decile_table(means)

    participant_summary.to_csv(TABLE_DIR / "nonlinearity_participantwise.csv", index=False)
    stimulus_summary.to_csv(TABLE_DIR / "nonlinearity_stimulus_level.csv", index=False)
    deciles.to_csv(TABLE_DIR / "nonlinearity_artificiality_deciles.csv", index=False)
    mf.save_figure(mf.figure_nonlinearity(means), "nonlinearity_artificiality_trends", FIGURE_DIR)

    print("Participant-wise quadratic test (coefficients vs zero)")
    print(participant_summary.to_string(index=False))
    print("\nStimulus-level linear vs quadratic fits and extreme-quintile contrast")
    print(stimulus_summary.to_string(index=False))
    print("\nArtificiality decile means")
    print(deciles.to_string(index=False))
    return participant_summary, stimulus_summary, deciles


if __name__ == "__main__":
    run()
