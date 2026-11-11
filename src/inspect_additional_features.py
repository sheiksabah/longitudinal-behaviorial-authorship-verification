
from pathlib import Path
import pandas as pd

INPUT_FILE = Path("data/processed/additional_chunk_features.csv")

FEATURES = [
    "average_word_length",
    "hapax_ratio",
    "function_word_frequency_per_1000_words",
    "pronoun_frequency_per_1000_words",
    "article_frequency_per_1000_words",
    "conjunction_frequency_per_1000_words",
    "preposition_frequency_per_1000_words",
    "question_frequency_per_1000_words",
    "exclamation_frequency_per_1000_words"
]


def main():
    df = pd.read_csv(INPUT_FILE)

    for feature in FEATURES:
        print(f"\n{'=' * 70}")
        print(f"Feature: {feature}")
        print("=" * 70)

        print(
            df.nlargest(5, feature)[
                ["story", "chunk", "word_count", feature]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
