
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = Path(
    "data/processed/expanded_behavioral_dataset.csv"
)

OUTPUT_DIR = Path("data/processed")
PLOT_DIR = OUTPUT_DIR / "plots"

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


def main():
    df = pd.read_csv(INPUT_FILE)

    correlation = df[FEATURES].corr(method="pearson")

    correlation.to_csv(
        OUTPUT_DIR / "expanded_feature_correlations.csv"
    )

    fig, ax = plt.subplots(figsize=(15, 12))

    image = ax.imshow(
        correlation,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    ax.set_xticks(range(len(FEATURES)))
    ax.set_yticks(range(len(FEATURES)))

    ax.set_xticklabels(
        FEATURES,
        rotation=60,
        ha="right",
        fontsize=8
    )

    ax.set_yticklabels(FEATURES, fontsize=8)

    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            ax.text(
                j, i,
                f"{correlation.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7
            )

    ax.set_title("Correlation Between 15 Writing Features")

    fig.colorbar(image, ax=ax, label="Pearson Correlation")

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "expanded_feature_correlations.png",
        dpi=300
    )

    plt.close()

    print("Expanded correlation analysis complete!")
    print("\nCorrelation matrix:")
    print(correlation.round(3).to_string())

    print("\nSaved:")
    print(OUTPUT_DIR / "expanded_feature_correlations.csv")
    print(PLOT_DIR / "expanded_feature_correlations.png")


if __name__ == "__main__":
    main()
