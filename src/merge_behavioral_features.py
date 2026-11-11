
from pathlib import Path
import pandas as pd

ORIGINAL_FILE = Path(
    "data/processed/chunk_features_filtered.csv"
)

ADDITIONAL_FILE = Path(
    "data/processed/additional_chunk_features_filtered.csv"
)

OUTPUT_FILE = Path(
    "data/processed/expanded_behavioral_dataset.csv"
)


def main():
    original = pd.read_csv(ORIGINAL_FILE)
    additional = pd.read_csv(ADDITIONAL_FILE)

    # Match observations using story and chunk identifiers.
    merged = pd.merge(
        original,
        additional,
        on=["story", "chunk", "word_count"],
        how="inner",
        validate="one_to_one"
    )

    merged.to_csv(OUTPUT_FILE, index=False)

    print("Feature merging complete!")
    print(f"Original dataset: {len(original)} rows")
    print(f"Additional dataset: {len(additional)} rows")
    print(f"Merged dataset: {len(merged)} rows")
    print(f"Total columns: {len(merged.columns)}")

    print("\nColumns:")
    print(merged.columns.tolist())

    print("\nMissing values:")
    print(merged.isnull().sum().to_string())

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

