"""Smoke tests for the asset risk-class assignment."""

import pandas as pd

from src.profile_coherence.risk_classification import (
    AGGRESSIVE,
    BALANCED,
    CONSERVATIVE,
    INCOME,
    build_asset_risk_classes,
)


def _make_close_prices(returns_per_asset: dict[str, list[float]]) -> pd.DataFrame:
    """Build a tall close-prices DataFrame from per-asset daily multiplicative paths."""
    rows: list[dict[str, object]] = []
    for isin, returns in returns_per_asset.items():
        price = 100.0
        date_index = pd.date_range("2018-01-01", periods=len(returns), freq="B")
        for timestamp, multiplier in zip(date_index, returns, strict=True):
            price *= multiplier
            rows.append({"ISIN": isin, "timestamp": timestamp, "closePrice": price})
    return pd.DataFrame(rows)


def test_mtf_subcategory_overrides_volatility() -> None:
    """Mutual funds with a known subcategory should bypass the volatility-quartile branch."""
    asset_information = pd.DataFrame(
        [
            {
                "ISIN": "MTF_MM",
                "assetCategory": "MTF",
                "assetSubCategory": "Money Market",
            },
            {"ISIN": "MTF_BAL", "assetCategory": "MTF", "assetSubCategory": "Balanced"},
            {"ISIN": "MTF_EQ", "assetCategory": "MTF", "assetSubCategory": "Equity"},
        ]
    )
    close_prices = _make_close_prices({"MTF_MM": [1.0] * 300})

    classes = build_asset_risk_classes(asset_information, close_prices)

    assert classes["MTF_MM"] == CONSERVATIVE
    assert classes["MTF_BAL"] == BALANCED
    assert classes["MTF_EQ"] == AGGRESSIVE


def test_bond_subcategory_mapping() -> None:
    """Government bonds map to Conservative, corporate to Income, others to Income."""
    asset_information = pd.DataFrame(
        [
            {
                "ISIN": "BOND_G",
                "assetCategory": "Bond",
                "assetSubCategory": "Government",
            },
            {
                "ISIN": "BOND_C",
                "assetCategory": "Bond",
                "assetSubCategory": "Corporate",
            },
            {"ISIN": "BOND_X", "assetCategory": "Bond", "assetSubCategory": None},
        ]
    )
    close_prices = _make_close_prices({"BOND_G": [1.0] * 10})

    classes = build_asset_risk_classes(asset_information, close_prices)

    assert classes["BOND_G"] == CONSERVATIVE
    assert classes["BOND_C"] == INCOME
    assert classes["BOND_X"] == INCOME


def test_stock_volatility_quartiles_span_all_bands() -> None:
    """Four stocks with distinct volatility levels should each land in a different band."""
    import math

    daily_log_returns = {
        "STOCK_LOW": [math.exp(0.001)] * 300,
        "STOCK_MID_LOW": [
            math.exp(0.01) if i % 2 else math.exp(-0.01) for i in range(300)
        ],
        "STOCK_MID_HIGH": [
            math.exp(0.03) if i % 2 else math.exp(-0.03) for i in range(300)
        ],
        "STOCK_HIGH": [
            math.exp(0.08) if i % 2 else math.exp(-0.08) for i in range(300)
        ],
    }

    asset_information = pd.DataFrame(
        [
            {"ISIN": isin, "assetCategory": "Stock", "assetSubCategory": "Other"}
            for isin in daily_log_returns
        ]
    )
    close_prices = _make_close_prices(daily_log_returns)

    classes = build_asset_risk_classes(asset_information, close_prices)

    assert classes["STOCK_LOW"] < classes["STOCK_MID_LOW"]
    assert classes["STOCK_MID_LOW"] < classes["STOCK_MID_HIGH"]
    assert classes["STOCK_MID_HIGH"] < classes["STOCK_HIGH"]
    assert classes["STOCK_LOW"] == CONSERVATIVE
    assert classes["STOCK_HIGH"] == AGGRESSIVE
