"""Sequence-length distribution analysis for FAR-Trans user purchase sequences.

Sequential recommenders (SASRec, TiSASRec, HybridDualHead) attend over the
last `max_sequence_length` items of each user's chronological buy history.
Their ability to learn temporal structure depends on how many of those
positions actually carry signal, which in turn depends on how many buys
each user has. This module computes that distribution from the raw
transactions file and prints/saves summary statistics plus an optional
histogram.

Run via `poe analyze-sequences` (or
`uv run python -m src.data.sequence_analysis`).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src.config.settings import DataPaths
from src.data.loading import load_transactions
from src.data.sequences import build_user_sequences


@dataclass(frozen=True)
class SequenceLengthSummary:
    """Summary statistics for user purchase-sequence lengths."""

    number_of_users: int
    min_length: int
    max_length: int
    mean_length: float
    median_length: float
    percentile_25: float
    percentile_75: float
    percentile_90: float
    percentile_95: float
    percentile_99: float
    share_users_at_most_1: float
    share_users_at_most_3: float
    share_users_at_most_5: float
    share_users_at_most_10: float
    share_users_at_most_20: float
    share_users_at_most_50: float


def compute_sequence_lengths(
    buy_transactions: pd.DataFrame,
    cutoff_date: pd.Timestamp | None = None,
) -> pd.Series:
    """Return a Series of sequence lengths (one row per user).

    If cutoff_date is None, uses the full transaction history. Otherwise,
    only counts transactions strictly before cutoff_date (matching the
    sequence-construction convention in `src/data/sequences.py`).
    """
    if cutoff_date is None:
        cutoff = pd.Timestamp(buy_transactions["timestamp"].max()) + pd.Timedelta(
            days=1
        )
    else:
        cutoff = pd.Timestamp(cutoff_date)

    sequences = build_user_sequences(buy_transactions, cutoff.date())
    return pd.Series(
        {user_id: len(sequence) for user_id, sequence in sequences.items()},
        name="sequence_length",
    )


def summarise_sequence_lengths(lengths: pd.Series) -> SequenceLengthSummary:
    """Compute summary statistics from a Series of per-user sequence lengths."""
    number_of_users = len(lengths)

    def share_at_most(threshold: int) -> float:
        if number_of_users == 0:
            return 0.0
        return float((lengths <= threshold).sum()) / number_of_users

    return SequenceLengthSummary(
        number_of_users=number_of_users,
        min_length=int(lengths.min()) if number_of_users else 0,
        max_length=int(lengths.max()) if number_of_users else 0,
        mean_length=float(lengths.mean()) if number_of_users else 0.0,
        median_length=float(lengths.median()) if number_of_users else 0.0,
        percentile_25=float(lengths.quantile(0.25)) if number_of_users else 0.0,
        percentile_75=float(lengths.quantile(0.75)) if number_of_users else 0.0,
        percentile_90=float(lengths.quantile(0.90)) if number_of_users else 0.0,
        percentile_95=float(lengths.quantile(0.95)) if number_of_users else 0.0,
        percentile_99=float(lengths.quantile(0.99)) if number_of_users else 0.0,
        share_users_at_most_1=share_at_most(1),
        share_users_at_most_3=share_at_most(3),
        share_users_at_most_5=share_at_most(5),
        share_users_at_most_10=share_at_most(10),
        share_users_at_most_20=share_at_most(20),
        share_users_at_most_50=share_at_most(50),
    )


def format_summary(summary: SequenceLengthSummary) -> str:
    """Format a SequenceLengthSummary as a human-readable multi-line string."""
    lines = [
        "Sequence Length Distribution (buy transactions per user):",
        f"  Users with at least one buy : {summary.number_of_users:>10,}",
        f"  Min length                   : {summary.min_length:>10,}",
        f"  Max length                   : {summary.max_length:>10,}",
        f"  Mean length                  : {summary.mean_length:>10.2f}",
        f"  Median length                : {summary.median_length:>10.2f}",
        f"  25th percentile              : {summary.percentile_25:>10.2f}",
        f"  75th percentile              : {summary.percentile_75:>10.2f}",
        f"  90th percentile              : {summary.percentile_90:>10.2f}",
        f"  95th percentile              : {summary.percentile_95:>10.2f}",
        f"  99th percentile              : {summary.percentile_99:>10.2f}",
        "",
        "Cumulative share of users by sequence length cap:",
        f"  <=  1 transaction           : {summary.share_users_at_most_1:>10.2%}",
        f"  <=  3 transactions          : {summary.share_users_at_most_3:>10.2%}",
        f"  <=  5 transactions          : {summary.share_users_at_most_5:>10.2%}",
        f"  <= 10 transactions          : {summary.share_users_at_most_10:>10.2%}",
        f"  <= 20 transactions          : {summary.share_users_at_most_20:>10.2%}",
        f"  <= 50 transactions          : {summary.share_users_at_most_50:>10.2%}",
    ]
    return "\n".join(lines)


def save_histogram(
    lengths: pd.Series,
    output_path: Path,
    bin_cap: int = 50,
) -> None:
    """Save a histogram PNG of sequence lengths, capping the x-axis at bin_cap.

    Lengths greater than bin_cap are lumped into the final bin so the long
    tail does not flatten the body of the distribution. Matplotlib is a
    declared project dependency so this import is safe at call time.
    """
    import matplotlib.pyplot as plt

    capped = lengths.clip(upper=bin_cap)
    figure, axes = plt.subplots(figsize=(9, 5))
    axes.hist(capped, bins=range(1, bin_cap + 2), edgecolor="black", linewidth=0.5)
    axes.set_xlabel(f"Sequence length (clipped at {bin_cap})")
    axes.set_ylabel("Number of users")
    axes.set_title("User purchase-sequence length distribution (FAR-Trans)")
    axes.grid(axis="y", alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_summary_json(summary: SequenceLengthSummary, output_path: Path) -> None:
    """Save the summary as a JSON artifact for inclusion in reports."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(summary), indent=2))


def run_analysis(
    data_paths: DataPaths | None = None,
    output_directory: Path = Path("outputs/analysis/sequence_lengths"),
    bin_cap: int = 50,
) -> SequenceLengthSummary:
    """Load raw transactions, compute sequence-length distribution, save artifacts."""
    data_paths = data_paths or DataPaths()
    transactions_path = data_paths.data_directory / data_paths.transactions_file

    print(f"Loading transactions from {transactions_path}...")
    transactions = load_transactions(transactions_path)
    buy_transactions = transactions[transactions["transactionType"] == "Buy"].copy()
    print(f"  Loaded {len(buy_transactions):,} buy transactions")

    lengths = compute_sequence_lengths(buy_transactions)
    summary = summarise_sequence_lengths(lengths)

    print()
    print(format_summary(summary))

    histogram_path = output_directory / "histogram.png"
    summary_path = output_directory / "summary.json"

    save_histogram(lengths, histogram_path, bin_cap=bin_cap)
    save_summary_json(summary, summary_path)

    print()
    print(f"Histogram saved to {histogram_path}")
    print(f"Summary JSON saved to {summary_path}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyse FAR-Trans user purchase-sequence length distribution"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/analysis/sequence_lengths",
        help="Directory for histogram PNG and summary JSON",
    )
    parser.add_argument(
        "--bin-cap",
        type=int,
        default=50,
        help="Clip the histogram x-axis at this sequence length",
    )
    args = parser.parse_args()

    run_analysis(
        output_directory=Path(args.output_dir),
        bin_cap=args.bin_cap,
    )
