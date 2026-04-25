"""Smoke tests for the customer profile lookup."""

import pandas as pd

from src.profile_coherence.customer_profile import build_customer_profile_lookup
from src.profile_coherence.risk_classification import (
    AGGRESSIVE,
    BALANCED,
    CONSERVATIVE,
)


def test_declared_levels_round_trip_to_ordinal_bands() -> None:
    """Declared MiFID levels map to the four-band ordinal scale without flagging as predicted."""
    customer_information = pd.DataFrame(
        [
            {
                "customerID": "C1",
                "customerType": "Mass",
                "riskLevel": "Conservative",
                "investmentCapacity": "CAP_LT30K",
            },
            {
                "customerID": "C2",
                "customerType": "Premium",
                "riskLevel": "Aggressive",
                "investmentCapacity": "CAP_GT300K",
            },
        ]
    )

    profiles = build_customer_profile_lookup(customer_information)

    assert profiles["C1"].risk_band == CONSERVATIVE
    assert profiles["C1"].risk_band_is_predicted is False
    assert profiles["C2"].risk_band == AGGRESSIVE
    assert profiles["C2"].customer_type == "Premium"
    assert profiles["C2"].investment_capacity == "CAP_GT300K"


def test_predicted_levels_carry_flag() -> None:
    """`Predicted_*` risk levels map to the same ordinal but flag as imputed."""
    customer_information = pd.DataFrame(
        [
            {
                "customerID": "C1",
                "customerType": "Mass",
                "riskLevel": "Predicted_Balanced",
                "investmentCapacity": "Predicted_CAP_30K_80K",
            },
        ]
    )

    profiles = build_customer_profile_lookup(customer_information)

    assert profiles["C1"].risk_band == BALANCED
    assert profiles["C1"].risk_band_is_predicted is True
    assert profiles["C1"].investment_capacity == "Predicted_CAP_30K_80K"


def test_not_available_collapses_to_none() -> None:
    """`Not_Available` cells become `None` for all three signals."""
    customer_information = pd.DataFrame(
        [
            {
                "customerID": "C1",
                "customerType": "Not_Available",
                "riskLevel": "Not_Available",
                "investmentCapacity": "Not_Available",
            },
        ]
    )

    profiles = build_customer_profile_lookup(customer_information)

    assert profiles["C1"].risk_band is None
    assert profiles["C1"].customer_type is None
    assert profiles["C1"].investment_capacity is None
