import numpy as np
import pandas as pd
from statsmodels.genmod.cov_struct import Exchangeable, Independence
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE

from common import FIGURE_DIR, TABLE_DIR, ensure_output_dirs, load_data, validate_data
import manuscript_figures as mf


FORMULAS = {
    "trustworthiness": "Trustworthiness ~ Artificiality + Attractiveness",
    "attractiveness": "Attractiveness ~ Artificiality + Trustworthiness",
}


def fit_model(df, formula, covariance):
    return GEE.from_formula(
        formula,
        groups="participant_id",
        data=df,
        family=Gaussian(),
        cov_struct=covariance,
    ).fit()


def coefficient_table(result, outcome, working_correlation):
    ci = result.conf_int()
    return pd.DataFrame(
        {
            "outcome": outcome,
            "working_correlation": working_correlation,
            "term": result.params.index,
            "beta": result.params.values,
            "robust_SE": result.bse.values,
            "z": result.tvalues.values,
            "p": result.pvalues.values,
            "CI95_low": ci.iloc[:, 0].values,
            "CI95_high": ci.iloc[:, 1].values,
        }
    )


def run():
    ensure_output_dirs()
    df = load_data().sort_values(["participant_id", "session", "stimulus_id"])
    validate_data(df)

    results = {}
    tables = []
    for outcome, formula in FORMULAS.items():
        for label, covariance in (
            ("Exchangeable", Exchangeable()),
            ("Independent", Independence()),
        ):
            result = fit_model(df, formula, covariance)
            results[(outcome, label)] = result
            tables.append(coefficient_table(result, outcome, label))

    table = pd.concat(tables, ignore_index=True)
    table.to_csv(TABLE_DIR / "gee_coefficients.csv", index=False)
    mf.save_figure(mf.figure_gee_predictions(df), "gee_marginal_predictions", FIGURE_DIR)
    print(table.to_string(index=False))
    return results, table


if __name__ == "__main__":
    run()
