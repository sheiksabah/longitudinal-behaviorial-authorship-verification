
from pathlib import Path
import pandas as pd

INPUT_FILE = Path("data/processed/chunk_features_filtered.csv")
OUTPUT_DIR = Path("data/processed")

FEATURES = [
    "average_sentence_length",
    "sentence_length_std",
    "type_token_ratio",
    "comma_frequency_per_1000_words",
    "semicolon_frequency_per_1000_words"
]

MIN_CHUNKS = 4


def main():
    df = pd.read_csv(INPUT_FILE)

    results = []

    for story, group in df.groupby("story"):
        # Correlations are unreliable with very few chunks.
        if len(group) < MIN_CHUNKS:
            continue

        correlation = group[FEATURES].corr()

        for i, feature_a in enumerate(FEATURES):
            for feature_b in FEATURES[i + 1:]:
                results.append({
                    "story": story,
                    "chunk_count": len(group),
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "correlation": correlation.loc[feature_a, feature_b]
                })

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR / "within_story_correlations.csv",
        index=False
    )

    # Summarize relationships across eligible stories.
    summary = (
        results_df.groupby(["feature_a", "feature_b"])["correlation"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR / "within_story_correlation_summary.csv",
        index=False
    )

    print("Within-story correlation analysis complete!")
    print(f"Stories analyzed: {results_df['story'].nunique()}")
    print("\nCorrelation summary:")
    print(summary.round(3).to_string(index=False))

    print("\nSaved:")
    print("data/processed/within_story_correlations.csv")
    print("data/processed/within_story_correlation_summary.csv")


if __name__ == "__main__":
    main()


