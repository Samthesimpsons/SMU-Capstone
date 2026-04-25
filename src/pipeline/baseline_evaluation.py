"""Ray-driven grid sweep over RF and LightGCN, with the post-evaluation decomposition wired in."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import ray
import torch
from ray import tune

from src.config.schemas import EvaluationResult, TemporalSplitData
from src.config.settings import (
    DataPaths,
    ExperimentConfig,
    LightGCNConfig,
    ModelConfig,
    RandomForestConfig,
)
from src.data.loading import load_assets, load_customers
from src.evaluation.metrics import (
    CALENDAR_DAYS_PER_MONTH,
    build_price_lookup,
    evaluate_model_on_split,
)
from src.features.technical_indicators import build_indicator_dataframe
from src.models.light_gcn import LightGCNBaseline
from src.models.protocol import Recommender
from src.models.random_forest import RandomForestBaseline
from src.pipeline.preprocessing import (
    load_evaluation_splits,
    load_preprocessed_close_prices,
)
from src.profile_coherence.customer_profile import build_customer_profile_lookup
from src.profile_coherence.risk_classification import build_asset_risk_classes

PrimaryMetric = Literal["ndcg", "roi", "recall", "profile_coherence"]

_PRIMARY_METRIC_TO_KEY: dict[PrimaryMetric, str] = {
    "ndcg": "average_ndcg",
    "roi": "average_roi",
    "recall": "average_recall",
    "profile_coherence": "average_profile_coherence",
}


RANDOM_FOREST_GRID: dict[str, list[Any]] = {
    "number_of_estimators": [20, 50, 100],
    "max_depth": [None, 15],
    "random_state": [42],
    "prediction_horizon_months": [6],
}


LIGHT_GCN_GRID: dict[str, list[Any]] = {
    "embedding_dimension": [64, 128],
    "number_of_layers": [2, 3],
    "learning_rate": [1e-2, 1e-3],
    "weight_decay": [1e-5],
    "keep_probability": [0.6],
    "number_of_epochs": [50],
    "batch_size": [1024],
}


@dataclass(frozen=True)
class GridSpec:
    """Static description of one model's tuning sweep."""

    model_name: str
    config_class: type[ModelConfig]
    grid: dict[str, list[Any]]
    primary_metric: PrimaryMetric
    needs_indicators: bool
    use_gpu_per_trial: bool
    max_concurrent_trials: int


GRID_SPECS: dict[str, GridSpec] = {
    "random_forest": GridSpec(
        model_name="random_forest",
        config_class=RandomForestConfig,
        grid=RANDOM_FOREST_GRID,
        primary_metric="roi",
        needs_indicators=True,
        use_gpu_per_trial=False,
        max_concurrent_trials=3,
    ),
    "light_gcn": GridSpec(
        model_name="light_gcn",
        config_class=LightGCNConfig,
        grid=LIGHT_GCN_GRID,
        primary_metric="ndcg",
        needs_indicators=False,
        use_gpu_per_trial=True,
        max_concurrent_trials=4,
    ),
}


@dataclass(frozen=True)
class EvaluationContext:
    """Inputs every trial needs to train and score on the 69 evaluation splits."""

    splits: list[TemporalSplitData]
    close_prices: pd.DataFrame
    indicator_dataframe: pd.DataFrame | None
    customer_profiles: dict[str, Any]
    asset_risk_classes: dict[str, int]
    top_k: int
    device: str
    project_root: Path
    run_timestamp: str


def _set_random_seeds(seed: int) -> None:
    """Seed numpy, torch, and Python random for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(preferred_device: str) -> str:
    """Return the requested device, falling back to CPU when CUDA is unavailable."""
    if preferred_device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return preferred_device


def _generate_recommendations(
    model: Recommender, split: TemporalSplitData, k: int
) -> dict[str, list[str]]:
    """Top-k recommendations for every eligible customer in `split`."""
    return {
        customer_id: model.recommend_for_user(
            customer_id,
            split.training_interactions.get(customer_id, set()),
            k,
        )
        for customer_id in split.eligible_customer_ids
    }


def _instantiate_model(
    spec: GridSpec, config: ModelConfig, context: EvaluationContext
) -> Recommender:
    """Map a spec + config to a concrete recommender instance."""
    if spec.model_name == "random_forest":
        assert isinstance(config, RandomForestConfig)
        assert context.indicator_dataframe is not None
        return RandomForestBaseline(
            random_forest_config=config,
            indicator_dataframe=context.indicator_dataframe,
        )
    if spec.model_name == "light_gcn":
        assert isinstance(config, LightGCNConfig)
        return LightGCNBaseline(config=config)
    raise ValueError(f"Unknown model: {spec.model_name}")


@dataclass(frozen=True)
class TrialEvaluation:
    """Outputs of a single trial: per-split metrics + the recommendations themselves."""

    per_split_results: list[EvaluationResult]
    recommendations_per_split: list[tuple[TemporalSplitData, dict[str, list[str]]]]


def _evaluate_over_splits(
    model: Recommender, context: EvaluationContext
) -> TrialEvaluation:
    """Train, evaluate, AND retain per-split recommendations for one trial."""
    results: list[EvaluationResult] = []
    recommendations_per_split: list[tuple[TemporalSplitData, dict[str, list[str]]]] = []
    for split in context.splits:
        model.train_on_split(split, device=context.device)
        recommendations = _generate_recommendations(model, split, context.top_k)
        result = evaluate_model_on_split(
            recommendations,
            split,
            context.close_prices,
            context.customer_profiles,
            context.asset_risk_classes,
            context.top_k,
        )
        results.append(result.model_copy(update={"model_name": model.name}))
        recommendations_per_split.append((split, recommendations))
    return TrialEvaluation(
        per_split_results=results,
        recommendations_per_split=recommendations_per_split,
    )


def _summarise(results: list[EvaluationResult]) -> dict[str, float]:
    """Reduce per-split results to four scalar averages."""
    if not results:
        return {
            "average_ndcg": 0.0,
            "average_roi": 0.0,
            "average_recall": 0.0,
            "average_profile_coherence": 0.0,
        }
    n = len(results)
    return {
        "average_ndcg": sum(r.ndcg_at_k for r in results) / n,
        "average_roi": sum(r.roi_at_k for r in results) / n,
        "average_recall": sum(r.recall_at_k for r in results) / n,
        "average_profile_coherence": sum(r.profile_coherence_at_k for r in results) / n,
    }


def _save_per_trial_metrics_csv(
    hyperparameters: dict[str, Any],
    trial_id: str,
    evaluation: TrialEvaluation,
    trial_directory: Path,
) -> None:
    """Save per-split scalar metrics for one trial to its own directory.

    Per-trial directories avoid the concurrent-append race the legacy single
    CSV had when multiple Ray workers wrote at once.
    """
    trial_directory.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "trial_id": trial_id,
            **hyperparameters,
            "split_index": result.split_index,
            "time_point": result.time_point.isoformat(),
            "ndcg_at_k": result.ndcg_at_k,
            "roi_at_k": result.roi_at_k,
            "recall_at_k": result.recall_at_k,
            "profile_coherence_at_k": result.profile_coherence_at_k,
        }
        for result in evaluation.per_split_results
    ]
    pd.DataFrame(rows).to_csv(trial_directory / "per_split_metrics.csv", index=False)


def _compute_monthly_return(
    start_price: float, end_price: float, days_in_period: int
) -> float:
    """Geometric monthly return matching the FAR-Trans ROI@k convention."""
    if start_price <= 0.0 or days_in_period <= 0:
        return 0.0
    total_return = (end_price - start_price) / start_price
    return pow(1.0 + total_return, CALENDAR_DAYS_PER_MONTH / days_in_period) - 1.0


def _save_recommendations_parquet(
    hyperparameters: dict[str, Any],
    trial_id: str,
    evaluation: TrialEvaluation,
    close_prices: pd.DataFrame,
    trial_directory: Path,
) -> None:
    """Save flat per-recommendation rows for one trial as parquet.

    Schema: (trial_id, <hyperparameters>, split_index, time_point, test_end,
    customer_id, rank, asset_id, monthly_return, is_relevant). Customer and
    asset bands are deliberately omitted: they are derivable from the same
    `customer_profiles` and `asset_risk_classes` lookup tables that the
    pipeline already loads, so leaving them out lets a future analysis swap
    risk-classification rules without re-running the GPU job.
    """
    trial_directory.mkdir(parents=True, exist_ok=True)

    columns: dict[str, list[Any]] = {"trial_id": []}
    for key in hyperparameters:
        columns[key] = []
    for column_name in (
        "split_index",
        "time_point",
        "test_end",
        "customer_id",
        "rank",
        "asset_id",
        "monthly_return",
        "is_relevant",
    ):
        columns[column_name] = []

    for split, recommendations in evaluation.recommendations_per_split:
        price_lookup = build_price_lookup(
            close_prices,
            split.time_point,
            split.test_end,
            split.eligible_asset_ids,
        )
        days_in_period = (split.test_end - split.time_point).days
        eligible_assets = set(split.eligible_asset_ids)
        time_point_iso = split.time_point.isoformat()
        test_end_iso = split.test_end.isoformat()

        for customer_id in split.eligible_customer_ids:
            relevant_assets = (
                split.test_interactions.get(customer_id, set()) & eligible_assets
            )
            customer_recs = recommendations.get(customer_id, [])
            for rank_index, asset_id in enumerate(customer_recs, start=1):
                start_price, end_price = price_lookup.get(asset_id, (0.0, 0.0))
                monthly_return = _compute_monthly_return(
                    start_price, end_price, days_in_period
                )
                columns["trial_id"].append(trial_id)
                for key, value in hyperparameters.items():
                    columns[key].append(value)
                columns["split_index"].append(split.split_index)
                columns["time_point"].append(time_point_iso)
                columns["test_end"].append(test_end_iso)
                columns["customer_id"].append(customer_id)
                columns["rank"].append(rank_index)
                columns["asset_id"].append(asset_id)
                columns["monthly_return"].append(monthly_return)
                columns["is_relevant"].append(asset_id in relevant_assets)

    pd.DataFrame(columns).to_parquet(
        trial_directory / "recommendations.parquet", index=False
    )


def _build_grid_search_space(spec: GridSpec) -> dict[str, Any]:
    """Wrap each axis in `tune.grid_search` so Ray expands the cartesian product."""
    return {name: tune.grid_search(values) for name, values in spec.grid.items()}


def _trial_resources(
    spec: GridSpec, use_gpu: bool, max_concurrent_trials: int
) -> dict[str, float]:
    """Allocate fractional GPU when requested; CPU only otherwise.

    `max_concurrent_trials` controls the GPU split: with N concurrent GPU
    trials each gets `1/N` of the device. CPU-only trials always claim 1 CPU.
    """
    if spec.use_gpu_per_trial and use_gpu:
        return {"gpu": 1.0 / max(1, max_concurrent_trials), "cpu": 1.0}
    return {"cpu": 1.0}


def _make_trainable(
    spec: GridSpec, context_ref: ray.ObjectRef, evaluation_dir: Path
) -> Callable[[dict[str, Any]], None]:
    """Return a Ray-tunable function that trains, evaluates, and persists outputs.

    Each trial writes both `per_split_metrics.csv` and `recommendations.parquet`
    to its own subdirectory. The shared evaluation context travels via Ray's
    object store rather than being captured directly.
    """

    def trainable(hyperparameters: dict[str, Any]) -> None:
        context: EvaluationContext = ray.get(context_ref)
        config = spec.config_class(**hyperparameters)
        _set_random_seeds(0)
        model = _instantiate_model(spec, config, context)
        evaluation = _evaluate_over_splits(model, context)
        averages = _summarise(evaluation.per_split_results)

        trial_id = tune.get_context().get_trial_id()
        trial_directory = evaluation_dir / trial_id
        _save_per_trial_metrics_csv(
            hyperparameters=hyperparameters,
            trial_id=trial_id,
            evaluation=evaluation,
            trial_directory=trial_directory,
        )
        _save_recommendations_parquet(
            hyperparameters=hyperparameters,
            trial_id=trial_id,
            evaluation=evaluation,
            close_prices=context.close_prices,
            trial_directory=trial_directory,
        )

        tune.report(averages)

    return trainable


def _run_grid(
    spec: GridSpec,
    context: EvaluationContext,
    use_gpu: bool,
    results_directory: Path,
    max_concurrent_trials_override: int | None,
) -> tuple[ModelConfig, tune.ResultGrid]:
    """Run one model's grid via Ray Tune and return the best config + the result grid."""
    # Ray workers run with a different cwd, so relative paths land in worker temp
    # dirs. Resolve to absolute before handing the path to the trainable.
    evaluation_dir = (
        results_directory / "evaluation" / spec.model_name / context.run_timestamp
    ).resolve()
    context_ref = ray.put(context)
    trainable = _make_trainable(spec, context_ref, evaluation_dir)
    effective_concurrency = (
        max_concurrent_trials_override
        if max_concurrent_trials_override is not None
        else spec.max_concurrent_trials
    )

    tuner = tune.Tuner(
        tune.with_resources(
            trainable, _trial_resources(spec, use_gpu, effective_concurrency)
        ),
        param_space=_build_grid_search_space(spec),
        tune_config=tune.TuneConfig(
            metric=_PRIMARY_METRIC_TO_KEY[spec.primary_metric],
            mode="max",
            num_samples=1,
            max_concurrent_trials=effective_concurrency,
        ),
        run_config=tune.RunConfig(
            name=f"{spec.model_name}_grid_search",
            verbose=1,
        ),
    )

    result_grid = tuner.fit()
    best_result = result_grid.get_best_result(
        metric=_PRIMARY_METRIC_TO_KEY[spec.primary_metric], mode="max"
    )
    assert best_result.config is not None and best_result.metrics is not None
    best_config = spec.config_class(**dict(best_result.config))
    return best_config, result_grid


def _save_trial_summary(
    spec: GridSpec,
    result_grid: tune.ResultGrid,
    results_directory: Path,
    timestamp: str,
) -> Path:
    """Save one row per trial with hyperparameters and the four averaged metrics."""
    rows: list[dict[str, Any]] = []
    for result in result_grid:
        if result.metrics is None or result.config is None:
            continue
        row: dict[str, Any] = dict(result.config)
        row["average_ndcg"] = result.metrics.get("average_ndcg")
        row["average_roi"] = result.metrics.get("average_roi")
        row["average_recall"] = result.metrics.get("average_recall")
        row["average_profile_coherence"] = result.metrics.get(
            "average_profile_coherence"
        )
        rows.append(row)

    output_directory = results_directory / "tuning" / spec.model_name
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{timestamp}.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def _save_best_configs(best_configs: dict[str, ModelConfig], output_path: Path) -> None:
    """Persist best-by-primary-metric configs to JSON for downstream weeks."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: config.model_dump() for name, config in best_configs.items()}
    output_path.write_text(json.dumps(payload, indent=2))


def _build_evaluation_context(
    splits_directory: Path,
    data_paths: DataPaths,
    needs_indicators: bool,
    splits_limit: int | None,
    top_k: int,
    device: str,
    project_root: Path,
    run_timestamp: str,
) -> EvaluationContext:
    """Load and bundle every input that the trainables need."""
    print("Loading evaluation splits...")
    splits = load_evaluation_splits(splits_directory)
    if splits_limit is not None:
        splits = splits[:splits_limit]
    print(f"  Loaded {len(splits)} evaluation splits")

    print("Loading close prices...")
    close_prices = load_preprocessed_close_prices(splits_directory)

    print("Loading customer profiles and asset risk classes...")
    customer_information = load_customers(
        data_paths.data_directory / data_paths.customer_information_file
    )
    customer_profiles = build_customer_profile_lookup(customer_information)
    asset_information = load_assets(
        data_paths.data_directory / data_paths.asset_information_file
    )
    asset_risk_classes = build_asset_risk_classes(asset_information, close_prices)

    indicator_dataframe: pd.DataFrame | None = None
    if needs_indicators:
        print("Building indicator DataFrame for Random Forest...")
        indicator_dataframe = build_indicator_dataframe(close_prices)
        print(f"  Built indicators: {len(indicator_dataframe)} rows")

    return EvaluationContext(
        splits=splits,
        close_prices=close_prices,
        indicator_dataframe=indicator_dataframe,
        customer_profiles=customer_profiles,
        asset_risk_classes=asset_risk_classes,
        top_k=top_k,
        device=device,
        project_root=project_root,
        run_timestamp=run_timestamp,
    )


def _ray_runtime_env(project_root: Path) -> dict[str, object]:
    """Ship only the source tree to Ray workers."""
    return {"py_modules": [str(project_root / "src")]}


def run_baseline_grid_search(
    splits_directory: Path,
    results_directory: Path,
    *,
    experiment_config: ExperimentConfig | None = None,
    data_paths: DataPaths | None = None,
    selected_models: list[str] | None = None,
    splits_limit: int | None = None,
    max_concurrent_trials_override: int | None = None,
) -> dict[str, ModelConfig]:
    """Run the grid sweep across RF + LightGCN, save outputs, return best configs.

    `max_concurrent_trials_override` lets callers force serialised execution
    when the host is memory-constrained (typical on local dev boxes). On the
    cluster this stays `None` so the spec defaults apply.
    """
    experiment_config = experiment_config or ExperimentConfig()
    data_paths = data_paths or DataPaths()
    requested = selected_models or list(GRID_SPECS.keys())
    unknown = set(requested) - set(GRID_SPECS.keys())
    if unknown:
        raise ValueError(f"Unknown model names: {sorted(unknown)}")

    project_root = Path(__file__).resolve().parents[2]
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    needs_indicators = any(GRID_SPECS[name].needs_indicators for name in requested)
    device = _resolve_device(experiment_config.device)

    context = _build_evaluation_context(
        splits_directory=splits_directory,
        data_paths=data_paths,
        needs_indicators=needs_indicators,
        splits_limit=splits_limit,
        top_k=experiment_config.top_k,
        device=device,
        project_root=project_root,
        run_timestamp=run_timestamp,
    )

    os.environ["RAY_RUNTIME_ENV_LOCAL_DEV_MODE"] = "1"
    os.environ["RAY_ENABLE_LOG_MONITOR"] = "0"
    ray.init(
        log_to_driver=False,
        logging_level="error",
        include_dashboard=False,
        _enable_object_reconstruction=False,
        object_store_memory=1_073_741_824,
        runtime_env=_ray_runtime_env(project_root),
    )

    best_configs: dict[str, ModelConfig] = {}
    use_gpu = device != "cpu"
    try:
        for model_name in requested:
            spec = GRID_SPECS[model_name]
            print(
                f"\n=== Running {spec.model_name} grid ({_grid_size(spec)} trials) ==="
            )
            best_config, result_grid = _run_grid(
                spec,
                context,
                use_gpu,
                results_directory,
                max_concurrent_trials_override,
            )
            summary_path = _save_trial_summary(
                spec, result_grid, results_directory, run_timestamp
            )
            best_configs[model_name] = best_config
            print(f"  Trial summary saved to {summary_path}")
    finally:
        ray.shutdown()

    best_config_path = (
        Path("outputs/configs") / run_timestamp / "best_hyperparameters.json"
    )
    _save_best_configs(best_configs, best_config_path)
    print(f"\nBest configs saved to {best_config_path}")
    print("\nBest configs by primary metric:")
    for name, config in best_configs.items():
        print(f"  {name}: {config}")

    # Run the post-evaluation decomposition automatically so the cluster job
    # produces the headline tables and scatter plots in the same invocation.
    # Imported here (rather than at module level) to keep import-time cheap and
    # to avoid a circular import between pipeline and analysis packages.
    from src.analysis.baseline_decomposition import run_decomposition

    print("\nRunning post-evaluation decomposition...")
    run_decomposition(
        evaluation_directory=results_directory / "evaluation",
        run_timestamp=run_timestamp,
        data_paths=data_paths,
        top_k=experiment_config.top_k,
    )
    return best_configs


def _grid_size(spec: GridSpec) -> int:
    """Return the cardinality of the cartesian product of the grid."""
    size = 1
    for values in spec.grid.values():
        size *= len(values)
    return size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Ray-driven grid search across RF + LightGCN. Each trial is a full "
            "69-split evaluation. Paper defaults are guaranteed to be one of "
            "the trial points in each grid."
        )
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default="data/splits",
        help="Directory containing preprocessed splits (default: data/splits)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="outputs/results",
        help="Directory to write per-trial CSVs (default: outputs/results)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Compute device: cuda or cpu (default: cuda)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(GRID_SPECS.keys()),
        default=None,
        help="Subset of {random_forest, light_gcn} (default: both)",
    )
    parser.add_argument(
        "--splits-limit",
        type=int,
        default=None,
        help="Run only the first N splits (smoke test). Default: all 69 splits.",
    )
    parser.add_argument(
        "--max-concurrent-trials",
        type=int,
        default=None,
        help=(
            "Force serialised / lower-concurrency Ray execution. Useful on "
            "memory-constrained hosts. Default: per-spec values from GRID_SPECS."
        ),
    )
    args = parser.parse_args()

    run_baseline_grid_search(
        splits_directory=Path(args.splits_dir),
        results_directory=Path(args.results_dir),
        experiment_config=ExperimentConfig(device=args.device),
        selected_models=args.models,
        splits_limit=args.splits_limit,
        max_concurrent_trials_override=args.max_concurrent_trials,
    )
