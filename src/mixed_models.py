import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import TABLE_DIR, ensure_output_dirs, load_data, validate_data


FORMULAS = {
    "trustworthiness": "Trustworthiness ~ Artificiality + Attractiveness",
    "attractiveness": "Attractiveness ~ Artificiality + Trustworthiness",
}


def fit_with_fallbacks(formula, data):
    attempts = []
    captured_warnings = []
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
                result = model.fit(reml=False, method=optimizer, maxiter=1000, disp=False)
                captured_warnings.extend(str(item.message) for item in warning_list)
                attempts.append((optimizer, "success", bool(result.converged)))
                if result.converged:
                    return result, attempts, captured_warnings
            except Exception as error:
                captured_warnings.extend(str(item.message) for item in warning_list)
                attempts.append((optimizer, f"failed: {error}", False))
    raise RuntimeError(f"Mixed model failed: {attempts}")


def fixed_effect_table(result, outcome):
    names = list(result.fe_params.index)
    ci = result.conf_int().loc[names]
    return pd.DataFrame(
        {
            "outcome": outcome,
            "term": names,
            "beta": result.fe_params.loc[names].values,
            "SE": result.bse.loc[names].values,
            "z": result.tvalues.loc[names].values,
            "p": result.pvalues.loc[names].values,
            "CI95_low": ci.iloc[:, 0].values,
            "CI95_high": ci.iloc[:, 1].values,
        }
    )


def model_information(result, outcome, attempts, warning_messages):
    participant_variance = float(result.cov_re.iloc[0, 0])
    stimulus_variance = float(result.vcomp[0])
    return {
        "outcome": outcome,
        "converged": bool(result.converged),
        "logLik": float(result.llf),
        "AIC": float(result.aic),
        "BIC": float(result.bic),
        "participant_intercept_variance": participant_variance,
        "stimulus_intercept_variance": stimulus_variance,
        "residual_variance": float(result.scale),
        "optimizer_attempts": repr(attempts),
        "warnings": " | ".join(sorted(set(warning_messages))),
    }


def run():
    ensure_output_dirs()
    df = load_data().sort_values(["participant_id", "stimulus_id"])
    validate_data(df)

    fixed_tables = []
    information = []
    results = {}
    for outcome, formula in FORMULAS.items():
        result, attempts, warning_messages = fit_with_fallbacks(formula, df)
        results[outcome] = result
        fixed_tables.append(fixed_effect_table(result, outcome))
        information.append(model_information(result, outcome, attempts, warning_messages))

    fixed = pd.concat(fixed_tables, ignore_index=True)
    info = pd.DataFrame(information)
    fixed.to_csv(TABLE_DIR / "cross_classified_fixed_effects.csv", index=False)
    info.to_csv(TABLE_DIR / "cross_classified_model_information.csv", index=False)
    print(fixed.to_string(index=False))
    print("\nModel information")
    print(info.to_string(index=False))
    return results, fixed, info


if __name__ == "__main__":
    run()

