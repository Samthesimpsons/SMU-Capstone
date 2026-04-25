"""Smoke tests for pairwise discordance and the coherence predicate."""

from src.profile_coherence.discordance import (
    compute_pairwise_discordance,
    is_profile_coherent,
)


def test_pairwise_discordance_is_absolute_distance() -> None:
    """Discordance is |b_u - b_i|; symmetric, non-negative, zero on match."""
    assert compute_pairwise_discordance(0, 0) == 0
    assert compute_pairwise_discordance(0, 3) == 3
    assert compute_pairwise_discordance(3, 0) == 3


def test_pairwise_discordance_squared_overpenalises_far_misses() -> None:
    """Squared variant scales as `d^2`."""
    assert compute_pairwise_discordance(0, 3, squared=True) == 9
    assert compute_pairwise_discordance(0, 1, squared=True) == 1


def test_missing_band_returns_none() -> None:
    """Either side missing a band yields `None` (no usable signal)."""
    assert compute_pairwise_discordance(None, 1) is None
    assert compute_pairwise_discordance(2, None) is None


def test_coherence_predicate_default_tolerance() -> None:
    """Default `is_profile_coherent` accepts d <= 1."""
    assert is_profile_coherent(0) is True
    assert is_profile_coherent(1) is True
    assert is_profile_coherent(2) is False
    assert is_profile_coherent(None) is False


def test_strict_coherence_requires_exact_match() -> None:
    """Strict mode requires exact band match."""
    assert is_profile_coherent(0, strict=True) is True
    assert is_profile_coherent(1, strict=True) is False
