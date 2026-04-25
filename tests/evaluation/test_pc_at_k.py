"""Smoke tests for the PC@k metric."""

from src.evaluation.metrics import compute_profile_coherence_at_k


def test_all_within_tolerance_yields_one() -> None:
    """Every recommended asset within one band of the user gives PC=1.0."""
    asset_risk_classes = {"A": 1, "B": 1, "C": 2}

    score = compute_profile_coherence_at_k(
        ranked_recommendations=["A", "B", "C"],
        customer_band=1,
        asset_risk_classes=asset_risk_classes,
        k=3,
    )

    assert score == 1.0


def test_partial_match_gives_fraction() -> None:
    """The metric is the fraction of top-k that lies within tolerance."""
    asset_risk_classes = {"A": 0, "B": 3, "C": 3}

    score = compute_profile_coherence_at_k(
        ranked_recommendations=["A", "B", "C"],
        customer_band=0,
        asset_risk_classes=asset_risk_classes,
        k=3,
    )

    assert score == 1.0 / 3


def test_unknown_asset_band_is_treated_as_discordant() -> None:
    """Recommendations to assets without a known band are treated as discordant."""
    asset_risk_classes = {"A": 0}

    score = compute_profile_coherence_at_k(
        ranked_recommendations=["A", "MISSING", "MISSING"],
        customer_band=0,
        asset_risk_classes=asset_risk_classes,
        k=3,
    )

    assert score == 1.0 / 3


def test_no_user_band_returns_zero() -> None:
    """Users without a declared band contribute 0 to the average."""
    asset_risk_classes = {"A": 1}

    score = compute_profile_coherence_at_k(
        ranked_recommendations=["A"],
        customer_band=None,
        asset_risk_classes=asset_risk_classes,
        k=1,
    )

    assert score == 0.0


def test_strict_mode_requires_exact_band_match() -> None:
    """Strict variant only counts d == 0."""
    asset_risk_classes = {"A": 1, "B": 2}

    score = compute_profile_coherence_at_k(
        ranked_recommendations=["A", "B"],
        customer_band=1,
        asset_risk_classes=asset_risk_classes,
        k=2,
        strict=True,
    )

    assert score == 0.5
