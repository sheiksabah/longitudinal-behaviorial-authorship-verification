
from pathlib import Path
import pandas as pd

INPUT_FILE = Path(
    "data/processed/expanded_behavioral_dataset.csv"
)

OUTPUT_FILE = Path(
    "data/processed/reduced_behavioral_dataset.csv"
)

FEATURES = [
    "average_sentence_length",
    "sentence_length_std",
    "type_token_ratio",
    "comma_frequency_per_1000_words",
    "semicolon_frequency_per_1000_words",
    "average_word_length",
    "word_length_std",
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

    identifiers = ["story", "chunk", "word_count"]

    reduced = df[identifiers + FEATURES].copy()

    reduced.to_csv(OUTPUT_FILE, index=False)

    print("Reduced behavioral dataset created!")
    print(f"Observations: {len(reduced)}")
    print(f"Behavioral features: {len(FEATURES)}")
    print(f"Total columns: {len(reduced.columns)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
