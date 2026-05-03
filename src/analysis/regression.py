"""Generic OLS-with-cluster-robust-SE helpers used by the project's regressions."""

from __future__ import annotations

from typing import Any

import pandas as pd
import statsmodels.formula.api as smf


def fit_ols_clustered(formula: str, data: pd.DataFrame, cluster_column: str) -> Any:
    """Fit an OLS with cluster-robust standard errors clustered on the named column."""
    cluster_codes = data[cluster_column].astype("category").cat.codes
    return smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": cluster_codes.values}
    )


def coefficients_dataframe(model_fit: Any) -> pd.DataFrame:
    """Return a tidy table of estimates, standard errors, t / p values, and 95% CIs."""
    confidence_intervals = model_fit.conf_int()
    confidence_intervals.columns = ["ci_lower", "ci_upper"]
    table = pd.DataFrame(
        {
            "term": model_fit.params.index,
            "estimate": model_fit.params.values,
            "std_error": model_fit.bse.values,
            "t_value": model_fit.tvalues.values,
            "p_value": model_fit.pvalues.values,
        }
    )
    return table.merge(
        confidence_intervals.reset_index().rename(columns={"index": "term"}),
        on="term",
        how="left",
    )
