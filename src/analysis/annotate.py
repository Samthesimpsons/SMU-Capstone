"""Attach profile-coherence columns to recommendation rows."""

from __future__ import annotations

from typing import Any

import pandas as pd


def annotate_recommendations(
    recommendations: pd.DataFrame,
    customer_profiles: dict[str, Any],
    asset_risk_classes: dict[str, int],
) -> pd.DataFrame:
    """Attach customer_band, asset_band, discordance, and coherence flags."""
    annotated = recommendations.copy()
    annotated["customer_band"] = annotated["customer_id"].map(
        lambda customer_id: (
            profile.risk_band
            if (profile := customer_profiles.get(customer_id)) is not None
            else None
        )
    )
    annotated["asset_band"] = annotated["asset_id"].map(asset_risk_classes)

    customer_band = annotated["customer_band"].astype("Float64")
    asset_band = annotated["asset_band"].astype("Float64")
    discordance = (customer_band - asset_band).abs()
    annotated["discordance"] = discordance
    annotated["is_coherent"] = (discordance <= 1).astype("boolean")
    annotated["is_strictly_coherent"] = (discordance == 0).astype("boolean")
    return annotated
