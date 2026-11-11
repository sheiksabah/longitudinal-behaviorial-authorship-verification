
from pathlib import Path
import pandas as pd

INPUT_FILE = Path("data/processed/additional_chunk_features.csv")
OUTPUT_FILE = Path("data/processed/additional_chunk_features_filtered.csv")

MIN_WORD_COUNT = 500


def main():
    df = pd.read_csv(INPUT_FILE)

    original_count = len(df)

    filtered_df = df[
        df["word_count"] >= MIN_WORD_COUNT
    ].copy()

    filtered_df.to_csv(OUTPUT_FILE, index=False)

    print("Additional feature filtering complete!")
    print(f"Original chunks: {original_count}")
    print(f"Retained chunks: {len(filtered_df)}")
    print(f"Excluded chunks: {original_count - len(filtered_df)}")

    print("\nFiltered feature summary:")
    print(
        filtered_df.drop(
            columns=["story", "chunk"]
        ).describe().round(3).to_string()
    )

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
