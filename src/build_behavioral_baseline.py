
from pathlib import Path
import pandas as pd

INPUT_FILE = Path("data/processed/chunk_features_filtered.csv")
OUTPUT_FILE = Path("data/processed/behavioral_baseline.csv")

FEATURES = [
    "average_sentence_length",
    "sentence_length_std",
    "type_token_ratio",
    "comma_frequency_per_1000_words",
    "semicolon_frequency_per_1000_words"
]


def main():
    df = pd.read_csv(INPUT_FILE)

    # Calculate feature statistics for each individual story.
    story_stats = df.groupby("story")[FEATURES].agg(
        ["mean", "std"]
    )

    # Calculate the author-level baseline using story-level statistics.
    baseline = pd.DataFrame({
        "baseline_mean": story_stats.xs("mean", axis=1, level=1).mean(),
        "between_story_std": story_stats.xs("mean", axis=1, level=1).std(),
        "within_story_std_mean": story_stats.xs("std", axis=1, level=1).mean()
    })

    baseline.index.name = "feature"
    baseline = baseline.reset_index()

    baseline["story_count"] = df["story"].nunique()
    baseline["chunk_count"] = len(df)

    baseline.to_csv(OUTPUT_FILE, index=False)

    print("Behavioral baseline created!")
    print(f"Stories analyzed: {df['story'].nunique()}")
    print(f"Chunks analyzed: {len(df)}")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nBaseline results:")
    print(baseline.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
