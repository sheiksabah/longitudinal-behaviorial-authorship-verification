
from pathlib import Path
from collections import Counter
import re
import pandas as pd

INPUT_DIR = Path("data/processed/chunks")
OUTPUT_FILE = Path("data/processed/additional_chunk_features.csv")

# Grammatical word categories.
ARTICLES = {"a", "an", "the"}

PRONOUNS = {
    "i", "me", "my", "mine", "we", "us", "our", "ours",
    "you", "your", "yours", "he", "him", "his",
    "she", "her", "hers", "it", "its",
    "they", "them", "their", "theirs",
    "this", "that", "these", "those"
}

CONJUNCTIONS = {
    "and", "but", "or", "nor", "for", "yet", "so",
    "although", "because", "since", "unless", "while",
    "whereas", "if", "whether", "though"
}

PREPOSITIONS = {
    "about", "above", "across", "after", "against", "along",
    "among", "around", "at", "before", "behind", "below",
    "beneath", "beside", "between", "beyond", "by", "down",
    "during", "for", "from", "in", "inside", "into", "near",
    "of", "off", "on", "onto", "out", "outside", "over",
    "through", "throughout", "to", "toward", "under",
    "until", "up", "upon", "with", "within", "without"
}

FUNCTION_WORDS = (
    ARTICLES | PRONOUNS | CONJUNCTIONS | PREPOSITIONS
)


def extract_features(text):
    """Extract additional lexical and grammatical features."""

    words = re.findall(r"\b[\w']+\b", text.lower())
    word_count = len(words)

    if word_count == 0:
        return {}

    word_counts = Counter(words)

    # Lexical features.
    word_lengths = [len(word) for word in words]

    average_word_length = sum(word_lengths) / word_count

    word_length_std = pd.Series(word_lengths).std(ddof=0)

    hapax_count = sum(
        1 for count in word_counts.values() if count == 1
    )

    hapax_ratio = hapax_count / word_count

    # Function-word frequencies per 1,000 words.
    function_word_count = sum(
        word_counts[word] for word in FUNCTION_WORDS
    )

    pronoun_count = sum(
        word_counts[word] for word in PRONOUNS
    )

    article_count = sum(
        word_counts[word] for word in ARTICLES
    )

    conjunction_count = sum(
        word_counts[word] for word in CONJUNCTIONS
    )

    preposition_count = sum(
        word_counts[word] for word in PREPOSITIONS
    )

    # Sentence-form punctuation frequencies.
    question_count = text.count("?")
    exclamation_count = text.count("!")

    return {
        "average_word_length": average_word_length,
        "word_length_std": word_length_std,
        "hapax_ratio": hapax_ratio,
        "function_word_frequency_per_1000_words":
            function_word_count / word_count * 1000,
        "pronoun_frequency_per_1000_words":
            pronoun_count / word_count * 1000,
        "article_frequency_per_1000_words":
            article_count / word_count * 1000,
        "conjunction_frequency_per_1000_words":
            conjunction_count / word_count * 1000,
        "preposition_frequency_per_1000_words":
            preposition_count / word_count * 1000,
        "question_frequency_per_1000_words":
            question_count / word_count * 1000,
        "exclamation_frequency_per_1000_words":
            exclamation_count / word_count * 1000
    }


def main():
    results = []

    for story_dir in sorted(INPUT_DIR.iterdir()):
        if not story_dir.is_dir():
            continue

        for chunk_file in sorted(story_dir.glob("*.txt")):
            text = chunk_file.read_text(encoding="utf-8")

            features = extract_features(text)

            if not features:
                continue

            results.append({
                "story": story_dir.name,
                "chunk": chunk_file.stem,
                "word_count": len(
                    re.findall(r"\b[\w']+\b", text.lower())
                ),
                **features
            })

    df = pd.DataFrame(results)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("Additional feature extraction complete!")
    print(f"Chunks processed: {len(df)}")
    print(f"Additional features: {len(df.columns) - 3}")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nFeature summary:")
    print(df.describe().round(3).to_string())


if __name__ == "__main__":
    main()
