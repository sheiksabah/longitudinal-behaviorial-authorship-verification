
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = Path("data/processed/research_dataset.csv")
OUTPUT_DIR = Path("data/processed/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(DATA_FILE)

    features = [
        "word_count",
        "sentence_count",
        "average_sentence_length",
        "sentence_length_std",
        "unique_word_count",
        "type_token_ratio",
        "comma_frequency_per_1000_words",
        "semicolon_frequency_per_1000_words",
    ]

    # Calculate pairwise correlations between numerical features.
    correlation = df[features].corr()

    print("Feature correlation matrix:")
    print(correlation.round(2).to_string())

    # Create the heatmap using Matplotlib.
    plt.figure(figsize=(11, 8))

    plt.imshow(
        correlation,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        aspect="auto",
    )

    plt.colorbar(label="Correlation")

    plt.xticks(
        range(len(features)),
        features,
        rotation=45,
        ha="right",
    )
    plt.yticks(range(len(features)), features)

    # Display the correlation value in each cell.
    for i in range(len(features)):
        for j in range(len(features)):
            plt.text(
                j,
                i,
                f"{correlation.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    plt.title("Correlation Between Writing Features")
    plt.tight_layout()

    output = OUTPUT_DIR / "feature_correlation_heatmap.png"
    plt.savefig(output, dpi=150)
    plt.close()

    print(f"\nSaved heatmap to {output}")


if __name__ == "__main__":
    main()
