
from pathlib import Path
import re
import csv
import statistics

CORPUS_DIR = Path("data/processed/corpus")
OUTPUT_FILE = Path("data/processed/features.csv")


def load_story(path):
    # Corpus files have already been extracted and cleaned.
    return path.read_text(encoding="utf-8").strip()


def extract_features(text):
    words = re.findall(r"\b[\w']+\b", text.lower())

    # Approximate sentence segmentation.
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    word_counts = [
        len(re.findall(r"\b[\w']+\b", sentence))
        for sentence in sentences
    ]
    word_counts = [count for count in word_counts if count > 0]

    total_words = len(words)
    unique_words = len(set(words))

    return {
        "word_count": total_words,
        "sentence_count": len(word_counts),
        "average_sentence_length": round(
            statistics.mean(word_counts), 2
        ) if word_counts else 0,
        "sentence_length_std": round(
            statistics.pstdev(word_counts), 2
        ) if len(word_counts) > 1 else 0,
        "unique_word_count": unique_words,
        "type_token_ratio": round(
            unique_words / total_words, 4
        ) if total_words else 0,
        "comma_frequency_per_1000_words": round(
            text.count(",") / total_words * 1000, 2
        ) if total_words else 0,
        "semicolon_frequency_per_1000_words": round(
            text.count(";") / total_words * 1000, 2
        ) if total_words else 0,
    }


def main():
    rows = []

    for path in sorted(CORPUS_DIR.glob("*.txt")):
        text = load_story(path)
        features = extract_features(text)
        features["filename"] = path.name
        rows.append(features)

        print(f"{path.name}: {features['word_count']} words")

    if len(rows) != 22:
        raise ValueError(
            f"Expected 22 stories, but processed {len(rows)}."
        )

    fieldnames = ["filename"] + [
        key for key in rows[0] if key != "filename"
    ]

    with OUTPUT_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nProcessed {len(rows)} stories.")
    print(f"Saved features to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
