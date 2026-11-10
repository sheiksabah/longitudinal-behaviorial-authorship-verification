
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE = Path("data/processed/chunk_features_filtered.csv")
OUTPUT_DIR = Path("data/processed/plots")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_feature_distributions(df):
    """Visualize feature distributions across all chunks."""

    features = [
        ("average_sentence_length", "Average Sentence Length"),
        ("type_token_ratio", "Type-Token Ratio"),
        ("comma_frequency_per_1000_words", "Comma Frequency"),
        ("semicolon_frequency_per_1000_words", "Semicolon Frequency")
    ]

    for column, title in features:
        plt.figure(figsize=(9, 5))

        sns.boxplot(data=df, y=column)

        plt.title(f"{title} Across Poe's Writing Chunks")
        plt.ylabel(title)
        plt.tight_layout()

        filename = column + "_boxplot.png"
        plt.savefig(OUTPUT_DIR / filename, dpi=300)
        plt.close()


def plot_within_story_variation(df):
    """Compare average sentence length across individual stories."""

    plt.figure(figsize=(12, 9))

    sns.boxplot(
        data=df,
        x="average_sentence_length",
        y="story"
    )

    plt.title("Sentence Length Variation Within Individual Stories")
    plt.xlabel("Average Sentence Length (Words)")
    plt.ylabel("Story")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "sentence_length_by_story.png",
        dpi=300
    )
    plt.close()


def main():
    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded {len(df)} chunks from {df['story'].nunique()} stories.")

    plot_feature_distributions(df)
    plot_within_story_variation(df)

    print("\nVisualizations generated successfully!")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
