
import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/reduced_behavioral_dataset.csv")
OUTPUT = Path("data/processed/compact_behavioral_dataset.csv")

df = pd.read_csv(INPUT)

selected_features = [
    "average_sentence_length",
    "sentence_length_std",
    "type_token_ratio",
    "average_word_length",
    "word_length_std",
    "pronoun_frequency_per_1000_words",
    "comma_frequency_per_1000_words",
    "semicolon_frequency_per_1000_words"
]

identifiers = ["story", "chunk", "word_count"]

compact_df = df[identifiers + selected_features]

compact_df.to_csv(OUTPUT, index=False)

print("Compact dataset created!")
print("Observations:", len(compact_df))
print("Features:", len(selected_features))
print("\nSelected features:")
for feature in selected_features:
    print("-", feature)

print("\nSaved to:", OUTPUT)
