
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

FEATURES_FILE = Path("data/processed/features.csv")
OUTPUT_DIR = Path("data/processed/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(FEATURES_FILE)

    # Use shorter labels for chart readability.
    df["story"] = df["filename"].str.replace(
        ".txt", "", regex=False
    ).str.replace("_", " ").str.title()

    # Chart 1: Average sentence length.
    plt.figure(figsize=(9, 5))
    plt.bar(df["story"], df["average_sentence_length"])
    plt.title("Average Sentence Length by Story")
    plt.xlabel("Story")
    plt.ylabel("Words per Sentence")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "average_sentence_length.png", dpi=150)
    plt.close()

    # Chart 2: Type-token ratio.
    plt.figure(figsize=(9, 5))
    plt.bar(df["story"], df["type_token_ratio"])
    plt.title("Type-Token Ratio by Story")
    plt.xlabel("Story")
    plt.ylabel("Unique Words / Total Words")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "type_token_ratio.png", dpi=150)
    plt.close()

    # Chart 3: Punctuation frequencies.
    punctuation_columns = [
        "comma_frequency_per_1000_words",
        "semicolon_frequency_per_1000_words",
    ]

    punctuation = df.set_index("story")[punctuation_columns]
    punctuation.columns = ["Commas", "Semicolons"]

    punctuation.plot(kind="bar", figsize=(9, 5))
    plt.title("Punctuation Frequency by Story")
    plt.xlabel("Story")
    plt.ylabel("Occurrences per 1,000 Words")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "punctuation_frequency.png", dpi=150)
    plt.close()

    print(f"Charts saved to: {OUTPUT_DIR}")
    print("Created:")
    for path in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  {path}")


if __name__ == "__main__":
    main()

