import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats

from common import TABLE_DIR, ensure_output_dirs, load_data, validate_data


ZERO_ORDER = {
    "rho_art_att": ("Artificiality", "Attractiveness", "artificiality-attractiveness"),
    "rho_art_tru": ("Artificiality", "Trustworthiness", "artificiality-trustworthiness"),
    "rho_att_tru": ("Attractiveness", "Trustworthiness", "attractiveness-trustworthiness"),
}

PARTIAL = {
    "p_art_tru_given_att": (
        "Artificiality",
        "Trustworthiness",
        "Attractiveness",
        "artificiality-trustworthiness | attractiveness",
    ),
    "p_art_att_given_tru": (
        "Artificiality",
        "Attractiveness",
        "Trustworthiness",
        "artificiality-attractiveness | trustworthiness",
    ),
    "p_att_tru_given_art": (
        "Attractiveness",
        "Trustworthiness",
        "Artificiality",
        "attractiveness-trustworthiness | artificiality",
    ),
}


def participant_correlations(df):
    zero_rows = []
    partial_rows = []

    for participant_id, group in df.groupby("participant_id", sort=True):
        zero = {"participant_id": participant_id}
        for output, (x, y, _) in ZERO_ORDER.items():
            zero[output] = stats.spearmanr(group[x], group[y]).statistic
        zero_rows.append(zero)

        partial = {"participant_id": participant_id}
        for output, (x, y, covariate, _) in PARTIAL.items():
            result = pg.partial_corr(
                data=group,
                x=x,
                y=y,
                covar=covariate,
                method="spearman",
            )
            partial[output] = float(result["r"].iloc[0])
        partial_rows.append(partial)

    return pd.DataFrame(zero_rows), pd.DataFrame(partial_rows)


def fisher_summary(values, metric, family_size=3):
    r = pd.Series(values, dtype=float).dropna().to_numpy()
    r = np.clip(r, -0.999999, 0.999999)
    z = np.arctanh(r)
    n = len(z)
    mean_z = float(z.mean())
    sd_z = float(z.std(ddof=1))
    se_z = sd_z / np.sqrt(n)
    test = stats.ttest_1samp(z, popmean=0.0)

    def interval(confidence):
        alpha = 1.0 - confidence
        critical = stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)
        low = mean_z - critical * se_z
        high = mean_z + critical * se_z
        return np.tanh(low), np.tanh(high)

    ci95 = interval(0.95)
    adjusted_confidence = 1.0 - 0.05 / family_size
    ci_adjusted = interval(adjusted_confidence)
    p = float(test.pvalue)

    return {
        "metric": metric,
        "n": n,
        "mean_r": float(np.tanh(mean_z)),
        "CI95_r_low": float(ci95[0]),
        "CI95_r_high": float(ci95[1]),
        "CI98_33_r_low": float(ci_adjusted[0]),
        "CI98_33_r_high": float(ci_adjusted[1]),
        "t": float(test.statistic),
        "df": n - 1,
        "p": p,
        "p_bonf": min(p * family_size, 1.0),
        "significant_bonf": min(p * family_size, 1.0) < 0.05,
    }


def run():
    ensure_output_dirs()
    df = load_data()
    validate_data(df)
    corr_df, partial_df = participant_correlations(df)

    zero_summary = pd.DataFrame(
        [fisher_summary(corr_df[column], label) for column, (_, _, label) in ZERO_ORDER.items()]
    )
    partial_summary = pd.DataFrame(
        [
            fisher_summary(partial_df[column], label)
            for column, (_, _, _, label) in PARTIAL.items()
        ]
    )

    corr_df.to_csv(TABLE_DIR / "participant_zero_order_spearman.csv", index=False)
    partial_df.to_csv(TABLE_DIR / "participant_partial_spearman.csv", index=False)
    zero_summary.to_csv(TABLE_DIR / "fisherz_zero_order_summary.csv", index=False)
    partial_summary.to_csv(TABLE_DIR / "fisherz_partial_summary.csv", index=False)

    print("Zero-order Fisher z summary")
    print(zero_summary.to_string(index=False))
    print("\nPartial-correlation Fisher z summary")
    print(partial_summary.to_string(index=False))
    return corr_df, partial_df, zero_summary, partial_summary


if __name__ == "__main__":
    run()

