
from pathlib import Path
import pandas as pd

INPUT_FILE = Path(
    "data/processed/reduced_behavioral_dataset.csv"
)

OUTPUT_FILE = Path(
    "data/processed/standardized_behavioral_dataset.csv"
)

IDENTIFIERS = ["story", "chunk", "word_count"]

def main():
    df = pd.read_csv(INPUT_FILE)

    feature_columns = [
        column for column in df.columns
        if column not in IDENTIFIERS
    ]

    standardized = df.copy()

    # Calculate Z-scores for each feature.
    for feature in feature_columns:
        mean = df[feature].mean()
        std = df[feature].std(ddof=0)

        standardized[feature] = (
            df[feature] - mean
        ) / std

    standardized.to_csv(OUTPUT_FILE, index=False)

    print("Feature standardization complete!")
    print(f"Observations: {len(standardized)}")
    print(f"Features standardized: {len(feature_columns)}")

    print("\nStandardized feature means:")
    print(standardized[feature_columns].mean().round(4))

    print("\nStandardized feature standard deviations:")
    print(standardized[feature_columns].std(ddof=0).round(4))

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
