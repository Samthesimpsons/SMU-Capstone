"""Generate temporal evaluation splits and persist them to disk."""

import json
from pathlib import Path

import pandas as pd

from src.config.schemas import TemporalSplitData
from src.config.settings import DataPaths
from src.data.loading import load_close_prices, load_transactions
from src.data.splitting import generate_all_splits


def _save_split(split: TemporalSplitData, path: Path) -> None:
    """Serialize a single TemporalSplitData to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(split.model_dump_json(indent=2))


def _load_split(path: Path) -> TemporalSplitData:
    """Deserialize a single TemporalSplitData from JSON."""
    return TemporalSplitData.model_validate_json(path.read_text())


def save_splits(splits: list[TemporalSplitData], directory: Path) -> None:
    """Save a list of splits to numbered JSON files in a directory."""
    directory.mkdir(parents=True, exist_ok=True)
    for split in splits:
        filename = f"split_{split.split_index:03d}.json"
        _save_split(split, directory / filename)


def load_splits(directory: Path) -> list[TemporalSplitData]:
    """Load all split JSON files from a directory, sorted by split index."""
    split_files = sorted(directory.glob("split_*.json"))
    return [_load_split(path) for path in split_files]


def load_evaluation_splits(splits_root: Path) -> list[TemporalSplitData]:
    """Load evaluation splits from the preprocessing output directory."""
    return load_splits(splits_root / "evaluation")


def load_preprocessed_close_prices(splits_root: Path) -> pd.DataFrame:
    """Load close prices using the path stored in preprocessing metadata."""
    metadata_path = splits_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    close_prices_path = Path(metadata["close_prices_path"])
    return load_close_prices(close_prices_path)


def _save_metadata(
    output_directory: Path,
    close_prices_path: Path,
    number_of_evaluation_splits: int,
) -> None:
    """Save preprocessing metadata to JSON."""
    metadata = {
        "number_of_evaluation_splits": number_of_evaluation_splits,
        "close_prices_path": str(close_prices_path),
    }
    (output_directory / "metadata.json").write_text(json.dumps(metadata, indent=2))


def run_preprocessing(
    output_directory: Path,
    data_paths: DataPaths | None = None,
) -> None:
    """Load raw data, generate evaluation splits, and save them to disk."""
    data_paths = data_paths or DataPaths()

    print("Loading raw data...")
    transactions_path = data_paths.data_directory / data_paths.transactions_file
    close_prices_path = data_paths.data_directory / data_paths.close_prices_file
    transactions = load_transactions(transactions_path)
    close_prices = load_close_prices(close_prices_path)

    print("Generating evaluation splits...")
    evaluation_splits = generate_all_splits(transactions, close_prices)
    save_splits(evaluation_splits, output_directory / "evaluation")
    print(f"  Saved {len(evaluation_splits)} evaluation splits")

    _save_metadata(output_directory, close_prices_path, len(evaluation_splits))

    unique_customers: set[str] = set()
    unique_assets: set[str] = set()
    for split in evaluation_splits:
        unique_customers.update(split.eligible_customer_ids)
        unique_assets.update(split.eligible_asset_ids)

    print(f"\nPreprocessing complete: {output_directory}/")
    print(f"  Evaluation splits: {len(evaluation_splits)}")
    print(f"  Unique eligible customers: {len(unique_customers)}")
    print(f"  Unique eligible assets: {len(unique_assets)}")


if __name__ == "__main__":
    run_preprocessing(output_directory=Path("data/splits"))
