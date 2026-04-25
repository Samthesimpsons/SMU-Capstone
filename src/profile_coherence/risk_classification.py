"""Map each asset to one of the four ordinal MiFID II risk bands."""

from __future__ import annotations

import numpy as np
import pandas as pd

CONSERVATIVE = 0
INCOME = 1
BALANCED = 2
AGGRESSIVE = 3

RISK_BAND_NAMES: dict[int, str] = {
    CONSERVATIVE: "Conservative",
    INCOME: "Income",
    BALANCED: "Balanced",
    AGGRESSIVE: "Aggressive",
}

NUMBER_OF_RISK_BANDS = 4

TRADING_DAYS_PER_YEAR = 252
DEFAULT_VOLATILITY_WINDOW = TRADING_DAYS_PER_YEAR

_MTF_SUBCATEGORY_BAND: dict[str, int] = {
    "Money Market": CONSERVATIVE,
    "Bond": INCOME,
    "Bonds": INCOME,
    "Balanced": BALANCED,
    "Equity": AGGRESSIVE,
    "Large Cap": AGGRESSIVE,
}

_BOND_SUBCATEGORY_BAND: dict[str, int] = {
    "Government": CONSERVATIVE,
    "Corporate": INCOME,
}


def _classify_by_subcategory(
    asset_category: str, asset_subcategory: str | None
) -> int | None:
    """Return the risk band implied by the asset's category metadata, or None."""
    if asset_subcategory is None or pd.isna(asset_subcategory):
        return INCOME if asset_category == "Bond" else None

    if asset_category == "MTF":
        return _MTF_SUBCATEGORY_BAND.get(asset_subcategory)

    if asset_category == "Bond":
        return _BOND_SUBCATEGORY_BAND.get(asset_subcategory, INCOME)

    return None


def compute_annualised_volatility(
    close_prices: pd.DataFrame,
    window: int = DEFAULT_VOLATILITY_WINDOW,
) -> pd.Series:
    """Compute the trailing-window annualised volatility for each asset.

    Returns a Series indexed by ISIN holding the most-recent rolling
    standard-deviation of daily log-returns scaled by sqrt(252). Assets
    with fewer than `window` observations are dropped.
    """
    sorted_prices = close_prices.sort_values(["ISIN", "timestamp"])
    log_returns = (
        sorted_prices.groupby("ISIN")["closePrice"]
        .apply(lambda group: np.log(group).diff())
        .reset_index(level=0, drop=True)
    )
    sorted_prices = sorted_prices.assign(log_return=log_returns)

    grouped = sorted_prices.groupby("ISIN")["log_return"]
    rolling_std = grouped.rolling(window=window, min_periods=window).std()

    annualised = rolling_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    annualised.index = annualised.index.droplevel(0)
    sorted_prices = sorted_prices.assign(annualised_volatility=annualised)

    latest_per_asset = (
        sorted_prices.dropna(subset=["annualised_volatility"])
        .groupby("ISIN")["annualised_volatility"]
        .last()
    )
    return pd.Series(latest_per_asset)


def _band_from_volatility_quartile(
    isin: str,
    volatility_lookup: pd.Series,
    quartile_cutoffs: tuple[float, float, float] | None,
) -> int:
    """Assign a risk band to a single asset by its volatility quartile."""
    if quartile_cutoffs is None or isin not in volatility_lookup.index:
        return BALANCED

    q1, q2, q3 = quartile_cutoffs
    asset_volatility = float(volatility_lookup.loc[isin])
    if asset_volatility <= q1:
        return CONSERVATIVE
    if asset_volatility <= q2:
        return INCOME
    if asset_volatility <= q3:
        return BALANCED
    return AGGRESSIVE


def build_asset_risk_classes(
    asset_information: pd.DataFrame,
    close_prices: pd.DataFrame,
    volatility_window: int = DEFAULT_VOLATILITY_WINDOW,
) -> dict[str, int]:
    """Return a mapping `asset_id -> ordinal MiFID risk band`.

    The volatility-quartile fallback is computed across the cross-section of
    *Stock* assets so that the four bands are roughly balanced for the
    universe most exposed to band ambiguity. Funds and bonds that match a
    deterministic sub-category mapping are assigned without using volatility.
    """
    volatility_lookup = compute_annualised_volatility(
        close_prices, window=volatility_window
    )

    stock_isins = set(
        asset_information.loc[asset_information["assetCategory"] == "Stock", "ISIN"]
    )
    stock_volatility = volatility_lookup[
        volatility_lookup.index.isin(stock_isins)
    ].dropna()

    quartile_cutoffs: tuple[float, float, float] | None = None
    if len(stock_volatility) >= 4:
        q1, q2, q3 = np.quantile(stock_volatility.to_numpy(), [0.25, 0.5, 0.75])
        quartile_cutoffs = (float(q1), float(q2), float(q3))

    risk_classes: dict[str, int] = {}
    for _, row in asset_information.iterrows():
        isin = str(row["ISIN"])
        category = str(row["assetCategory"])
        subcategory = row.get("assetSubCategory")
        subcategory_str = str(subcategory) if isinstance(subcategory, str) else None
        band_from_metadata = _classify_by_subcategory(category, subcategory_str)
        if band_from_metadata is not None:
            risk_classes[isin] = band_from_metadata
            continue
        risk_classes[isin] = _band_from_volatility_quartile(
            isin, volatility_lookup, quartile_cutoffs
        )

    return risk_classes
