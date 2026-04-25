"""Pairwise profile discordance and the coherence predicate."""

from __future__ import annotations

import pandas as pd

from src.config.schemas import CustomerProfile


def compute_pairwise_discordance(
    customer_band: int | None,
    asset_band: int | None,
    *,
    squared: bool = False,
) -> int | None:
    """Return the discordance between a customer profile and an asset risk class.

    Returns None if either side has no usable signal. The squared variant
    over-penalises far misses and is used as a sensitivity check on the
    learning-side regulariser.
    """
    if customer_band is None or asset_band is None:
        return None
    distance = abs(customer_band - asset_band)
    return distance * distance if squared else distance


def is_profile_coherent(
    discordance: int | None,
    *,
    strict: bool = False,
) -> bool:
    """Return True when discordance is within the coherence tolerance.

    `strict=False` (default): coherent when `d <= 1`. `strict=True`: coherent
    when `d == 0` (exact band match).
    """
    if discordance is None:
        return False
    return discordance == 0 if strict else discordance <= 1


def annotate_transactions_with_discordance(
    transactions: pd.DataFrame,
    customer_profiles: dict[str, CustomerProfile],
    asset_risk_classes: dict[str, int],
    *,
    squared: bool = False,
) -> pd.DataFrame:
    """Return a copy of `transactions` with discordance and band columns attached.

    The added columns are:
    - `customer_band` (int | NaN)
    - `asset_band` (int | NaN)
    - `discordance` (int | NaN), squared when `squared=True`
    - `is_profile_coherent` (bool); True iff `discordance <= 1` and both
       bands are present.
    """
    result = transactions.copy()
    customer_bands: list[float] = []
    asset_bands: list[float] = []
    discordances: list[float] = []
    coherence_flags: list[bool] = []

    for _, row in result.iterrows():
        customer_id = str(row["customerID"])
        asset_id = str(row["ISIN"])
        profile = customer_profiles.get(customer_id)
        customer_band = profile.risk_band if profile is not None else None
        asset_band = asset_risk_classes.get(asset_id)

        discordance = compute_pairwise_discordance(
            customer_band, asset_band, squared=squared
        )

        customer_bands.append(
            float(customer_band) if customer_band is not None else float("nan")
        )
        asset_bands.append(
            float(asset_band) if asset_band is not None else float("nan")
        )
        discordances.append(
            float(discordance) if discordance is not None else float("nan")
        )
        coherence_flags.append(is_profile_coherent(discordance))

    result["customer_band"] = customer_bands
    result["asset_band"] = asset_bands
    result["discordance"] = discordances
    result["is_profile_coherent"] = coherence_flags
    return result
