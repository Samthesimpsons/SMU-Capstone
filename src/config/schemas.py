"""Pydantic schemas for temporal splits, customer profiles, and evaluation results."""

from datetime import date

from pydantic import BaseModel


class TemporalSplitData(BaseModel):
    """A single temporal train/test split with filtered user and asset sets."""

    model_config = {"arbitrary_types_allowed": True}

    split_index: int
    time_point: date
    test_end: date
    training_interactions: dict[str, set[str]]
    test_interactions: dict[str, set[str]]
    eligible_customer_ids: list[str]
    eligible_asset_ids: list[str]
    customer_id_to_index: dict[str, int]
    asset_id_to_index: dict[str, int]


class CustomerProfile(BaseModel):
    """Regulatory profile signals available for a single customer.

    `risk_band` is the ordinal MiFID II level on a 4-band scale:
    Conservative=0, Income=1, Balanced=2, Aggressive=3. `None` means the
    customer has no usable risk signal (raw `Not_Available` in FAR-Trans).
    The `predicted_*` flag distinguishes regression-imputed values from
    questionnaire-declared ones, which the FAR-Trans paper documents as a
    fallback for customers who never completed the MiFID survey.
    """

    customer_id: str
    risk_band: int | None
    risk_band_is_predicted: bool
    customer_type: str | None
    investment_capacity: str | None


class EvaluationResult(BaseModel):
    """Evaluation metrics for a model on a single temporal split."""

    split_index: int
    time_point: date
    model_name: str
    ndcg_at_k: float
    roi_at_k: float
    recall_at_k: float
    profile_coherence_at_k: float
