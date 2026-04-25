"""nDCG@k, ROI@k, Recall@k, and PC@k computation."""

import math
from datetime import date

import pandas as pd

from src.config.schemas import CustomerProfile, EvaluationResult, TemporalSplitData
from src.profile_coherence.discordance import (
    compute_pairwise_discordance,
    is_profile_coherent,
)

CALENDAR_DAYS_PER_MONTH = 30


def compute_ndcg_at_k(
    ranked_recommendations: list[str],
    relevant_assets: set[str],
    k: int = 10,
) -> float:
    """Compute nDCG@k for a single user's recommendation list."""
    if not relevant_assets:
        return 0.0

    top_k = ranked_recommendations[:k]

    dcg = 0.0
    for i, asset_id in enumerate(top_k):
        if asset_id in relevant_assets:
            dcg += 1.0 / math.log2(i + 2)

    ideal_relevant_count = min(k, len(relevant_assets))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_relevant_count))

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def compute_recall_at_k(
    ranked_recommendations: list[str],
    relevant_assets: set[str],
    k: int = 10,
) -> float:
    """Compute Recall@k for a single user's recommendation list."""
    if not relevant_assets:
        return 0.0

    top_k = set(ranked_recommendations[:k])
    hits = len(top_k & relevant_assets)
    return hits / len(relevant_assets)


def compute_roi_at_k(
    ranked_recommendations: list[str],
    price_lookup: dict[str, tuple[float, float]],
    days_in_period: int,
    k: int = 10,
) -> float:
    """Compute ROI@k: average geometric monthly return of top-k recommended assets.

    Recommended assets without a valid price are imputed as a 0% return rather
    than skipped, matching FAR-Trans semantics in
    `metrics/kpi_evaluation_metric.py:30-32` and
    `metrics/kpi_monthly_evaluation_metric.py`.
    """
    top_k = ranked_recommendations[:k]
    if not top_k:
        return 0.0

    monthly_returns: list[float] = []

    for asset_id in top_k:
        if asset_id in price_lookup:
            start_price, end_price = price_lookup[asset_id]
            if start_price > 0.0:
                total_return = (end_price - start_price) / start_price
                monthly_return = (
                    math.pow(
                        1.0 + total_return,
                        CALENDAR_DAYS_PER_MONTH / days_in_period,
                    )
                    - 1.0
                )
                monthly_returns.append(monthly_return)
                continue
        monthly_returns.append(0.0)

    return sum(monthly_returns) / len(monthly_returns)


def compute_profile_coherence_at_k(
    ranked_recommendations: list[str],
    customer_band: int | None,
    asset_risk_classes: dict[str, int],
    k: int = 10,
    *,
    strict: bool = False,
) -> float:
    """Compute PC@k for a single user's recommendation list.

    PC@k = (1/k) * |{i in top_k : |b_user - b_asset_i| <= tolerance}|.
    With `strict=True` the tolerance is 0 (exact band match); otherwise it is 1.

    Returns 0.0 when the user has no declared MiFID band (so this metric only
    contributes from users with profile signal). Recommendations that point to
    assets without a known band are treated as discordant.
    """
    if customer_band is None:
        return 0.0

    top_k = ranked_recommendations[:k]
    if not top_k:
        return 0.0

    coherent_count = 0
    for asset_id in top_k:
        asset_band = asset_risk_classes.get(asset_id)
        discordance = compute_pairwise_discordance(customer_band, asset_band)
        if is_profile_coherent(discordance, strict=strict):
            coherent_count += 1

    return coherent_count / len(top_k)


def build_price_lookup(
    close_prices: pd.DataFrame,
    time_point: date,
    test_end: date,
    asset_ids: list[str],
) -> dict[str, tuple[float, float]]:
    """Build a mapping from ISIN to (price_at_start, price_at_end) for ROI computation.

    For each asset, find the closest available price on or before time_point and test_end.
    """
    time_point_timestamp = pd.Timestamp(time_point)
    test_end_timestamp = pd.Timestamp(test_end)

    grouped = close_prices.groupby("ISIN")
    lookup: dict[str, tuple[float, float]] = {}

    for asset_id in asset_ids:
        if asset_id not in grouped.groups:
            continue

        group = grouped.get_group(asset_id)
        assert isinstance(group, pd.DataFrame)
        asset_data = group.sort_values(by="timestamp")

        start_candidates = asset_data[asset_data["timestamp"] <= time_point_timestamp]
        end_candidates = asset_data[asset_data["timestamp"] <= test_end_timestamp]

        if start_candidates.empty or end_candidates.empty:
            continue

        start_price = float(start_candidates.iloc[-1]["closePrice"])
        end_price = float(end_candidates.iloc[-1]["closePrice"])
        lookup[asset_id] = (start_price, end_price)

    return lookup


def evaluate_model_on_split(
    recommendations: dict[str, list[str]],
    split: TemporalSplitData,
    close_prices: pd.DataFrame,
    customer_profiles: dict[str, CustomerProfile],
    asset_risk_classes: dict[str, int],
    k: int = 10,
) -> EvaluationResult:
    """Evaluate a model's recommendations on one temporal split.

    Averages nDCG@k, ROI@k, Recall@k, and PC@k across all eligible users.
    PC@k contributions are restricted to users with a declared MiFID band;
    users without a declared band contribute 0.0 (same convention as nDCG
    for users with no relevant items, so per-user denominators are aligned).
    """
    price_lookup = build_price_lookup(
        close_prices, split.time_point, split.test_end, split.eligible_asset_ids
    )

    days_in_period = (split.test_end - split.time_point).days
    if days_in_period <= 0:
        raise ValueError(
            f"Split {split.split_index}: test_end ({split.test_end}) is not after"
            f" time_point ({split.time_point})"
        )

    eligible_assets = set(split.eligible_asset_ids)
    ndcg_scores: list[float] = []
    roi_scores: list[float] = []
    recall_scores: list[float] = []
    pc_scores: list[float] = []

    for customer_id in split.eligible_customer_ids:
        customer_recommendations = recommendations.get(customer_id, [])
        relevant_assets = (
            split.test_interactions.get(customer_id, set()) & eligible_assets
        )
        profile = customer_profiles.get(customer_id)
        customer_band = profile.risk_band if profile is not None else None

        ndcg_scores.append(
            compute_ndcg_at_k(customer_recommendations, relevant_assets, k)
        )
        roi_scores.append(
            compute_roi_at_k(customer_recommendations, price_lookup, days_in_period, k)
        )
        recall_scores.append(
            compute_recall_at_k(customer_recommendations, relevant_assets, k)
        )
        pc_scores.append(
            compute_profile_coherence_at_k(
                customer_recommendations, customer_band, asset_risk_classes, k
            )
        )

    average_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    average_roi = sum(roi_scores) / len(roi_scores) if roi_scores else 0.0
    average_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    average_pc = sum(pc_scores) / len(pc_scores) if pc_scores else 0.0

    return EvaluationResult(
        split_index=split.split_index,
        time_point=split.time_point,
        model_name="",
        ndcg_at_k=average_ndcg,
        roi_at_k=average_roi,
        recall_at_k=average_recall,
        profile_coherence_at_k=average_pc,
    )
