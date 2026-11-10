
from pathlib import Path
import re
import pandas as pd
import numpy as np

CHUNKS_DIR = Path("data/processed/chunks")
OUTPUT_FILE = Path("data/processed/chunk_features.csv")


def extract_features(text):
    """Extract writing features from a text sample."""

    words = re.findall(r"\b[\w']+\b", text.lower())
    sentences = re.split(r"[.!?]+", text)

    sentences = [s.strip() for s in sentences if s.strip()]

    word_count = len(words)
    sentence_count = len(sentences)

    sentence_lengths = [
        len(re.findall(r"\b[\w']+\b", sentence))
        for sentence in sentences
    ]

    unique_words = set(words)

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "average_sentence_length": (
            np.mean(sentence_lengths) if sentence_lengths else 0
        ),
        "sentence_length_std": (
            np.std(sentence_lengths) if sentence_lengths else 0
        ),
        "unique_word_count": len(unique_words),
        "type_token_ratio": (
            len(unique_words) / word_count if word_count else 0
        ),
        "comma_frequency_per_1000_words": (
            text.count(",") / word_count * 1000 if word_count else 0
        ),
        "semicolon_frequency_per_1000_words": (
            text.count(";") / word_count * 1000 if word_count else 0
        )
    }


def main():
    records = []

    for story_dir in sorted(CHUNKS_DIR.iterdir()):

        if not story_dir.is_dir():
            continue

        for chunk_file in sorted(story_dir.glob("*.txt")):

            text = chunk_file.read_text(encoding="utf-8")

            features = extract_features(text)

            records.append({
                "story": story_dir.name,
                "chunk": chunk_file.stem,
                **features
            })

    df = pd.DataFrame(records)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Processed {len(df)} chunks.")
    print(f"Saved dataset to {OUTPUT_FILE}")
    print("\nDataset shape:", df.shape)
    print("\nFirst five rows:")
    print(df.head())


if __name__ == "__main__":
    main()
