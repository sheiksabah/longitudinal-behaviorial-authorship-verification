
from pathlib import Path
import pandas as pd

INPUT_FILE = Path("data/processed/chunk_features_filtered.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "within_story_variation.csv"

FEATURES = [
    "average_sentence_length",
    "sentence_length_std",
    "type_token_ratio",
    "comma_frequency_per_1000_words",
    "semicolon_frequency_per_1000_words"
]


def main():
    df = pd.read_csv(INPUT_FILE)

    # Calculate mean and standard deviation for each story
    grouped = df.groupby("story")[FEATURES].agg(["mean", "std"])

    # Flatten the multi-level column names
    grouped.columns = [
        f"{feature}_{stat}"
        for feature, stat in grouped.columns
    ]

    # Add the number of usable chunks per story
    grouped["chunk_count"] = df.groupby("story").size()

    # Save results
    grouped = grouped.reset_index()

    grouped.to_csv(OUTPUT_FILE, index=False)

    print("Within-story variation analysis complete!")
    print(f"Stories analyzed: {len(grouped)}")
    print(f"Total chunks analyzed: {len(df)}")
    print(f"Saved results to: {OUTPUT_FILE}")

    print("\nVariation results:")
    print(grouped.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
