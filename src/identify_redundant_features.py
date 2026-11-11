
from pathlib import Path
import pandas as pd

INPUT_FILE = Path(
    "data/processed/expanded_behavioral_dataset.csv"
)

OUTPUT_FILE = Path(
    "data/processed/highly_correlated_features.csv"
)

FEATURES = [
    "average_sentence_length",
    "sentence_length_std",
    "type_token_ratio",
    "comma_frequency_per_1000_words",
    "semicolon_frequency_per_1000_words",
    "average_word_length",
    "word_length_std",
    "hapax_ratio",
    "function_word_frequency_per_1000_words",
    "pronoun_frequency_per_1000_words",
    "article_frequency_per_1000_words",
    "conjunction_frequency_per_1000_words",
    "preposition_frequency_per_1000_words",
    "question_frequency_per_1000_words",
    "exclamation_frequency_per_1000_words"
]

THRESHOLD = 0.70


def main():
    df = pd.read_csv(INPUT_FILE)

    correlation = df[FEATURES].corr()

    results = []

    for i, feature_a in enumerate(FEATURES):
        for feature_b in FEATURES[i + 1:]:
            value = correlation.loc[feature_a, feature_b]

            if abs(value) >= THRESHOLD:
                results.append({
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "correlation": value
                })

    results_df = pd.DataFrame(results)

    results_df.to_csv(OUTPUT_FILE, index=False)

    print("Highly correlated feature pairs:")
    print(results_df.round(3).to_string(index=False))

    print(f"\nThreshold: {THRESHOLD}")
    print(f"Pairs identified: {len(results_df)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

