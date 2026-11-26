import argparse
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE

from common import (
    EXPRESSION_FEATURE_PATH,
    FIGURE_DIR,
    ROOT,
    TABLE_DIR,
    ensure_output_dirs,
    load_data,
    validate_data,
)
import manuscript_figures as mf


EXPRESSION_FEATURES = [
    "AU6",
    "AU12",
    "anger",
    "disgust",
    "fear",
    "happiness",
    "sadness",
    "surprise",
    "neutral",
]
STANDARDIZED_FEATURES = ["AU12", "AU6", "happiness"]
OUTCOMES = {
    "trustworthiness": {
        "outcome": "Trustworthiness",
        "covariate": "Attractiveness",
    },
    "attractiveness": {
        "outcome": "Attractiveness",
        "covariate": "Trustworthiness",
    },
}
GEE_ADJUSTMENTS = {
    "baseline": None,
    "AU12": "AU12_z",
    "AU6": "AU6_z",
    "happiness": "happiness_z",
}
LMM_ADJUSTMENTS = {"baseline": None, "AU12": "AU12_z"}
LMM_CACHE_DIR = ROOT / ".cache" / "expression_analysis"


def prepare_data():
    ratings = load_data()
    validate_data(ratings)
    features = pd.read_csv(EXPRESSION_FEATURE_PATH)
    required = {
        "stimulus_id",
        "n_faces_detected",
        "face_confidence",
        *EXPRESSION_FEATURES,
    }
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Expression feature file is missing: {sorted(missing)}")
    if len(features) != 500 or features["stimulus_id"].nunique() != 500:
        raise ValueError("Expression feature file must contain 500 unique stimuli.")
    if features[list(required)].isna().any().any():
        raise ValueError("Expression feature file contains missing analysis values.")

    for feature in STANDARDIZED_FEATURES:
        sd = features[feature].std(ddof=1)
        if not np.isfinite(sd) or sd == 0:
            raise ValueError(f"Cannot standardize {feature}: invalid standard deviation.")
        features[f"{feature}_z"] = (features[feature] - features[feature].mean()) / sd

    keep = ["stimulus_id", *[f"{name}_z" for name in STANDARDIZED_FEATURES]]
    data = ratings.merge(features[keep], on="stimulus_id", how="inner", validate="many_to_one")
    if len(data) != len(ratings):
        raise RuntimeError("Expression feature merge changed the number of rating rows.")
    return data.sort_values(["participant_id", "stimulus_id"]), features


def write_feature_summaries(data, features):
    descriptives = features[["face_confidence", *EXPRESSION_FEATURES]].describe().T
    descriptives.index.name = "feature"
    descriptives.reset_index().to_csv(
        TABLE_DIR / "expression_feature_descriptives.csv", index=False
    )

    stimulus_means = (
        data.groupby("stimulus_id")[["Artificiality", "Attractiveness", "Trustworthiness"]]
        .mean()
        .reset_index()
    )
    correlation_data = features[["stimulus_id", "AU6", "AU12", "happiness"]].merge(
        stimulus_means, on="stimulus_id", validate="one_to_one"
    )
    variables = [
        "AU6",
        "AU12",
        "happiness",
        "Artificiality",
        "Attractiveness",
        "Trustworthiness",
    ]
    rows = []
    for left_index, left in enumerate(variables):
        for right in variables[left_index + 1 :]:
            result = spearmanr(correlation_data[left], correlation_data[right])
            rows.append(
                {
                    "variable_1": left,
                    "variable_2": right,
                    "spearman_rho": float(result.statistic),
                    "p": float(result.pvalue),
                    "n_stimuli": len(correlation_data),
                }
            )
    pd.DataFrame(rows).to_csv(
        TABLE_DIR / "expression_rating_spearman_correlations.csv", index=False
    )


def coefficient_table(result, outcome, adjustment, analysis, se_name):
    ci = result.conf_int()
    return pd.DataFrame(
        {
            "analysis": analysis,
            "outcome": outcome,
            "expression_adjustment": adjustment,
            "term": result.params.index,
            "beta": result.params.values,
            se_name: result.bse.values,
            "z": result.tvalues.values,
            "p": result.pvalues.values,
            "CI95_low": ci.iloc[:, 0].values,
            "CI95_high": ci.iloc[:, 1].values,
        }
    )


def artificiality_comparison(table, se_column):
    art = table.loc[table["term"] == "Artificiality"].copy()
    rows = []
    for outcome, group in art.groupby("outcome", sort=False):
        baseline = float(
            group.loc[group["expression_adjustment"] == "baseline", "beta"].iloc[0]
        )
        for _, row in group.iterrows():
            beta = float(row["beta"])
            rows.append(
                {
                    "analysis": row["analysis"],
                    "outcome": outcome,
                    "expression_adjustment": row["expression_adjustment"],
                    "beta": beta,
                    se_column: row[se_column],
                    "z": row["z"],
                    "p": row["p"],
                    "CI95_low": row["CI95_low"],
                    "CI95_high": row["CI95_high"],
                    "beta_change_from_baseline": beta - baseline,
                    "percent_change_in_abs_beta": (
                        100.0 * (abs(beta) - abs(baseline)) / abs(baseline)
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_gee(data):
    fixed_tables = []
    information = []
    for outcome_label, specification in OUTCOMES.items():
        for adjustment, feature_z in GEE_ADJUSTMENTS.items():
            terms = ["Artificiality", specification["covariate"]]
            if feature_z is not None:
                terms.append(feature_z)
            formula = f"{specification['outcome']} ~ " + " + ".join(terms)
            result = GEE.from_formula(
                formula,
                groups="participant_id",
                data=data,
                family=Gaussian(),
                cov_struct=Exchangeable(),
            ).fit()
            fixed_tables.append(
                coefficient_table(
                    result,
                    outcome_label,
                    adjustment,
                    "GEE_exchangeable",
                    "robust_SE",
                )
            )
            information.append(
                {
                    "analysis": "GEE_exchangeable",
                    "outcome": outcome_label,
                    "expression_adjustment": adjustment,
                    "formula": formula,
                    "n_rows": int(result.nobs),
                    "n_participants": int(data["participant_id"].nunique()),
                    "working_correlation": float(
                        np.asarray(result.cov_struct.dep_params).squeeze()
                    ),
                    "converged": bool(result.converged),
                    "scale": float(result.scale),
                }
            )
            print(
                f"Fitted expression-adjusted GEE: {outcome_label} | {adjustment}",
                flush=True,
            )

    fixed = pd.concat(fixed_tables, ignore_index=True)
    info = pd.DataFrame(information)
    comparison = artificiality_comparison(fixed, "robust_SE")
    fixed.to_csv(TABLE_DIR / "expression_gee_fixed_effects.csv", index=False)
    info.to_csv(TABLE_DIR / "expression_gee_model_information.csv", index=False)
    comparison.to_csv(
        TABLE_DIR / "expression_gee_artificiality_comparison.csv", index=False
    )
    return fixed, info, comparison


def fit_lmm_with_fallbacks(formula, data):
    attempts = []
    captured_warnings = []
    last_result = None
    for optimizer in ("lbfgs", "powell", "cg"):
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            try:
                model = smf.mixedlm(
                    formula,
                    data=data,
                    groups=data["participant_id"],
                    re_formula="1",
                    vc_formula={"stimulus": "0 + C(stimulus_id)"},
                )
                result = model.fit(
                    reml=False, method=optimizer, maxiter=1000, disp=False
                )
                captured_warnings.extend(str(item.message) for item in warning_list)
                attempts.append((optimizer, "success", bool(result.converged)))
                last_result = result
                if result.converged:
                    return result, attempts, captured_warnings
            except Exception as error:
                captured_warnings.extend(str(item.message) for item in warning_list)
                attempts.append((optimizer, f"failed: {error}", False))
    if last_result is not None:
        return last_result, attempts, captured_warnings
    raise RuntimeError(f"Expression-adjusted mixed model failed: {attempts}")


def lmm_coefficient_table(result, outcome, adjustment):
    names = list(result.fe_params.index)
    ci = result.conf_int().loc[names]
    return pd.DataFrame(
        {
            "analysis": "cross_classified_LMM",
            "outcome": outcome,
            "expression_adjustment": adjustment,
            "term": names,
            "beta": result.fe_params.loc[names].values,
            "SE": result.bse.loc[names].values,
            "z": result.tvalues.loc[names].values,
            "p": result.pvalues.loc[names].values,
            "CI95_low": ci.iloc[:, 0].values,
            "CI95_high": ci.iloc[:, 1].values,
        }
    )


def run_lmm(data):
    LMM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fixed_tables = []
    information = []
    for outcome_label, specification in OUTCOMES.items():
        for adjustment, feature_z in LMM_ADJUSTMENTS.items():
            terms = ["Artificiality", specification["covariate"]]
            if feature_z is not None:
                terms.append(feature_z)
            formula = f"{specification['outcome']} ~ " + " + ".join(terms)
            print(
                f"Fitting expression-adjusted LMM: {outcome_label} | {adjustment}",
                flush=True,
            )
            result, attempts, warning_messages = fit_lmm_with_fallbacks(formula, data)
            fixed_tables.append(lmm_coefficient_table(result, outcome_label, adjustment))
            information.append(
                {
                    "analysis": "cross_classified_LMM",
                    "outcome": outcome_label,
                    "expression_adjustment": adjustment,
                    "formula": formula,
                    "converged": bool(result.converged),
                    "logLik": float(result.llf),
                    "AIC": float(result.aic),
                    "BIC": float(result.bic),
                    "participant_intercept_variance": float(result.cov_re.iloc[0, 0]),
                    "stimulus_intercept_variance": float(result.vcomp[0]),
                    "residual_variance": float(result.scale),
                    "optimizer_attempts": repr(attempts),
                    "warnings": " | ".join(sorted(set(warning_messages))),
                }
            )
            pd.concat(fixed_tables, ignore_index=True).to_csv(
                LMM_CACHE_DIR / "fixed_effects_checkpoint.csv", index=False
            )
            pd.DataFrame(information).to_csv(
                LMM_CACHE_DIR / "model_information_checkpoint.csv", index=False
            )
            print(
                f"Completed expression-adjusted LMM: {outcome_label} | "
                f"{adjustment} | converged={result.converged}",
                flush=True,
            )

    fixed = pd.concat(fixed_tables, ignore_index=True)
    info = pd.DataFrame(information)
    comparison = artificiality_comparison(fixed, "SE")
    fixed.to_csv(TABLE_DIR / "expression_lmm_fixed_effects.csv", index=False)
    info.to_csv(TABLE_DIR / "expression_lmm_model_information.csv", index=False)
    comparison.to_csv(
        TABLE_DIR / "expression_lmm_artificiality_comparison.csv", index=False
    )
    return fixed, info, comparison


def run(skip_lmm=False):
    ensure_output_dirs()
    data, features = prepare_data()
    write_feature_summaries(data, features)

    face_means = data.groupby("stimulus_id")[
        ["Artificiality", "Trustworthiness", "Attractiveness"]
    ].mean()
    mf.save_figure(
        mf.figure_au12_vs_ratings(face_means, features),
        "expression_au12_vs_ratings",
        FIGURE_DIR,
    )
    mf.save_figure(
        mf.figure_expression_robustness(data, features[["stimulus_id", "AU12"]]),
        "expression_partial_correlation_robustness",
        FIGURE_DIR,
    )

    gee_outputs = run_gee(data)
    lmm_outputs = None
    if skip_lmm:
        print("Skipped expression-adjusted LMMs by --skip-lmm.")
    else:
        lmm_outputs = run_lmm(data)
    print(f"Expression-analysis outputs written to {TABLE_DIR} and {FIGURE_DIR}")
    return gee_outputs, lmm_outputs


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the automated facial-expression robustness analyses."
    )
    parser.add_argument(
        "--skip-lmm",
        action="store_true",
        help="Skip the slower expression-adjusted cross-classified mixed models.",
    )
    args = parser.parse_args()
    run(skip_lmm=args.skip_lmm)


if __name__ == "__main__":
    main()
