
from pathlib import Path
import re
import pandas as pd
import numpy as np

# --------------------------------
# Paths
# --------------------------------

DATA_DIR = Path("data/longitudinal")
METADATA_FILE = DATA_DIR / "metadata/writing_samples.csv"
OUTPUT_FILE = DATA_DIR / "metadata/longitudinal_features.csv"

# --------------------------------
# Feature extraction
# --------------------------------

def extract_features(text):
    words = re.findall(r"\b[\w']+\b", text.lower())
    sentences = re.split(r"[.!?]+", text)

    sentences = [s.strip() for s in sentences if s.strip()]

    word_count = len(words)

    sentence_lengths = [
        len(re.findall(r"\b[\w']+\b", sentence))
        for sentence in sentences
    ]

    unique_words = set(words)

    word_lengths = [len(word) for word in words]

    pronouns = {
        "i", "me", "my", "mine", "we", "us", "our", "ours",
        "you", "your", "yours", "he", "him", "his",
        "she", "her", "hers", "it", "its", "they",
        "them", "their", "theirs", "this", "that",
        "these", "those"
    }

    return {
        "word_count": word_count,

        "average_sentence_length": (
            np.mean(sentence_lengths) if sentence_lengths else 0
        ),

        "sentence_length_std": (
            np.std(sentence_lengths) if sentence_lengths else 0
        ),

        "type_token_ratio": (
            len(unique_words) / word_count if word_count else 0
        ),

        "average_word_length": (
            np.mean(word_lengths) if word_lengths else 0
        ),

        "word_length_std": (
            np.std(word_lengths) if word_lengths else 0
        ),

        "pronoun_frequency_per_1000_words": (
            sum(word in pronouns for word in words) / word_count * 1000
            if word_count else 0
        ),

        "comma_frequency_per_1000_words": (
            text.count(",") / word_count * 1000 if word_count else 0
        ),

        "semicolon_frequency_per_1000_words": (
            text.count(";") / word_count * 1000 if word_count else 0
        )
    }


# --------------------------------
# Process longitudinal samples
# --------------------------------

def main():

    metadata = pd.read_csv(METADATA_FILE)

    records = []

    for _, sample in metadata.iterrows():

        filename = sample["filename"]

        if pd.isna(filename) or not str(filename).strip():
            print(f"Skipping {sample['sample_id']}: missing filename")
            continue

        file_path = DATA_DIR / str(filename)

        if not file_path.is_file():
            print(f"File not found: {file_path}")
            continue

        text = file_path.read_text(encoding="utf-8")

        features = extract_features(text)

        records.append({
            **sample.to_dict(),
            **features
        })

    result = pd.DataFrame(records)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print("\nLongitudinal feature extraction complete!")
    print("Samples processed:", len(result))
    print("Features extracted: 8")
    print("Saved to:", OUTPUT_FILE)

    if not result.empty:
        print("\nDataset preview:")
        print(result.head())


if __name__ == "__main__":
    main()
