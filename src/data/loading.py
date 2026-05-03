"""Raw dataset loading functions for the FAR-Trans CSV files."""

from pathlib import Path

import pandas as pd


def load_transactions(path: Path) -> pd.DataFrame:
    """Load the raw transactions CSV with timestamp parsing."""
    return pd.read_csv(path, parse_dates=["timestamp"])


def load_close_prices(path: Path) -> pd.DataFrame:
    """Load close prices, drop zero-price assets, deduplicate, and sort by (ISIN, timestamp)."""
    dataframe = pd.read_csv(path, parse_dates=["timestamp"])

    zero_price_assets = set(
        dataframe.loc[dataframe["closePrice"] == 0.0, "ISIN"].unique()
    )

    if zero_price_assets:
        dataframe = dataframe[~dataframe["ISIN"].isin(zero_price_assets)]

    dataframe = dataframe.drop_duplicates(subset=["ISIN", "timestamp"], keep="last")

    return dataframe.sort_values(["ISIN", "timestamp"]).reset_index(drop=True)


def load_customers(path: Path) -> pd.DataFrame:
    """Load customer information, deduplicated to the latest record per customer."""
    dataframe = pd.read_csv(path, parse_dates=["lastQuestionnaireDate", "timestamp"])
    dataframe = dataframe.sort_values("timestamp")
    return dataframe.drop_duplicates(subset="customerID", keep="last").reset_index(
        drop=True
    )


def load_assets(path: Path) -> pd.DataFrame:
    """Load asset information, deduplicated to the latest record per ISIN."""
    dataframe = pd.read_csv(path, parse_dates=["timestamp"])
    dataframe = dataframe.sort_values("timestamp")
    return dataframe.drop_duplicates(subset="ISIN", keep="last").reset_index(drop=True)
