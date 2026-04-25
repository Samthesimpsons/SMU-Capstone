"""Build per-customer CustomerProfile records from the FAR-Trans customer CSV."""

from __future__ import annotations

import pandas as pd

from src.config.schemas import CustomerProfile
from src.profile_coherence.risk_classification import (
    AGGRESSIVE,
    BALANCED,
    CONSERVATIVE,
    INCOME,
)

_DECLARED_RISK_LEVEL_TO_BAND: dict[str, int] = {
    "Conservative": CONSERVATIVE,
    "Income": INCOME,
    "Balanced": BALANCED,
    "Aggressive": AGGRESSIVE,
}

_PREDICTED_RISK_LEVEL_TO_BAND: dict[str, int] = {
    f"Predicted_{name}": band for name, band in _DECLARED_RISK_LEVEL_TO_BAND.items()
}


def _parse_risk_level(raw_value: object) -> tuple[int | None, bool]:
    """Map a raw `riskLevel` cell to (ordinal_band, is_predicted)."""
    if not isinstance(raw_value, str):
        return None, False
    if raw_value in _DECLARED_RISK_LEVEL_TO_BAND:
        return _DECLARED_RISK_LEVEL_TO_BAND[raw_value], False
    if raw_value in _PREDICTED_RISK_LEVEL_TO_BAND:
        return _PREDICTED_RISK_LEVEL_TO_BAND[raw_value], True
    return None, False


def _normalise_optional_string(raw_value: object) -> str | None:
    """Return None for missing or `Not_Available` cells, else the string value."""
    if not isinstance(raw_value, str):
        return None
    if raw_value == "Not_Available":
        return None
    return raw_value


def build_customer_profile_lookup(
    customer_information: pd.DataFrame,
) -> dict[str, CustomerProfile]:
    """Return a mapping `customer_id -> CustomerProfile`.

    `customer_information` is expected to be already deduplicated to the
    latest record per customer (handled by `src.data.loading.load_customers`).
    """
    profiles: dict[str, CustomerProfile] = {}
    for _, row in customer_information.iterrows():
        customer_id = str(row["customerID"])
        risk_band, is_predicted = _parse_risk_level(row.get("riskLevel"))
        profiles[customer_id] = CustomerProfile(
            customer_id=customer_id,
            risk_band=risk_band,
            risk_band_is_predicted=is_predicted,
            customer_type=_normalise_optional_string(row.get("customerType")),
            investment_capacity=_normalise_optional_string(
                row.get("investmentCapacity")
            ),
        )
    return profiles
