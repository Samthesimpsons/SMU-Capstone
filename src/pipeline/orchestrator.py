"""End-to-end thesis pipeline orchestrator."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config.settings import DataPaths, ExperimentConfig
from src.pipeline.decomposition import run_decomposition
from src.pipeline.grid_search import run_baseline_grid_search
from src.pipeline.panel_regression import run_panel_regression
from src.pipeline.transaction_return_regression import (
    run_transaction_return_regression,
)


def run_tune(
    splits_directory: Path = Path("data/splits"),
    results_directory: Path = Path("outputs/results"),
    *,
    experiment_config: ExperimentConfig | None = None,
    data_paths: DataPaths | None = None,
    splits_limit: int | None = None,
) -> None:
    """Run all four pipeline stages back-to-back."""
    experiment_config = experiment_config or ExperimentConfig()
    data_paths = data_paths or DataPaths()

    print("\n=== Stage 1: baseline grid sweep (RF + LightGCN) ===")
    _, baseline_run_timestamp = run_baseline_grid_search(
        splits_directory=splits_directory,
        results_directory=results_directory,
        experiment_config=experiment_config,
        data_paths=data_paths,
        splits_limit=splits_limit,
    )

    print("\n=== Stage 2: 2-model decomposition (RF + LightGCN) ===")
    run_decomposition(
        evaluation_directory=results_directory / "evaluation",
        run_timestamp=None,
        data_paths=data_paths,
        top_k=experiment_config.top_k,
    )

    print("\n=== Stage 3: RQ2 transaction-return regression ===")
    run_transaction_return_regression(data_paths=data_paths)

    print("\n=== Stage 4: RQ3 model panel regression (RF + LightGCN) ===")
    run_panel_regression(
        evaluation_directory=results_directory / "evaluation",
        data_paths=data_paths,
    )

    print(f"\nFull pipeline complete. Baseline run: {baseline_run_timestamp}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end thesis pipeline: baseline grid, 2-model decomposition, "
            "RQ2 transaction-return regression, and RQ3 model panel regression."
        )
    )
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--results-dir", type=str, default="outputs/results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--splits-limit", type=int, default=None)
    arguments = parser.parse_args()

    run_tune(
        splits_directory=Path(arguments.splits_dir),
        results_directory=Path(arguments.results_dir),
        experiment_config=ExperimentConfig(device=arguments.device),
        splits_limit=arguments.splits_limit,
    )
