
from pathlib import Path
import pandas as pd

INPUT_FILE = Path("data/processed/chunk_features.csv")
OUTPUT_FILE = Path("data/processed/chunk_features_filtered.csv")

MIN_WORD_COUNT = 500


def main():
    df = pd.read_csv(INPUT_FILE)

    original_count = len(df)

    # Keep chunks containing at least 500 words
    filtered_df = df[df["word_count"] >= MIN_WORD_COUNT].copy()

    excluded_count = original_count - len(filtered_df)

    # Save filtered dataset
    filtered_df.to_csv(OUTPUT_FILE, index=False)

    print("Chunk filtering complete!")
    print(f"Original chunks: {original_count}")
    print(f"Retained chunks: {len(filtered_df)}")
    print(f"Excluded chunks: {excluded_count}")
    print(f"Minimum word count: {MIN_WORD_COUNT}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
