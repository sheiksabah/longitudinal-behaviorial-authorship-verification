
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = Path("data/processed/chunk_features_filtered.csv")
OUTPUT_DIR = Path("data/processed")
PLOT_DIR = OUTPUT_DIR / "plots"

FEATURES = [
    "average_sentence_length",
    "sentence_length_std",
    "type_token_ratio",
    "comma_frequency_per_1000_words",
    "semicolon_frequency_per_1000_words"
]


def main():
    df = pd.read_csv(INPUT_FILE)

    # Calculate Pearson correlations between features.
    correlation_matrix = df[FEATURES].corr(method="pearson")

    print("Chunk-level correlation matrix:")
    print(correlation_matrix.round(3).to_string())

    # Save numerical results.
    correlation_matrix.to_csv(
        OUTPUT_DIR / "chunk_feature_correlations.csv"
    )

    # Visualize correlations.
    fig, ax = plt.subplots(figsize=(10, 8))

    image = ax.imshow(
        correlation_matrix,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    ax.set_xticks(range(len(FEATURES)))
    ax.set_yticks(range(len(FEATURES)))

    ax.set_xticklabels(FEATURES, rotation=45, ha="right")
    ax.set_yticklabels(FEATURES)

    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            ax.text(
                j, i,
                f"{correlation_matrix.iloc[i, j]:.2f}",
                ha="center",
                va="center"
            )

    ax.set_title("Correlation Between Writing Features")
    fig.colorbar(image, ax=ax, label="Pearson Correlation")

    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / "chunk_feature_correlations.png",
        dpi=300
    )
    plt.close()

    print("\nCorrelation analysis complete!")
    print("Saved CSV: data/processed/chunk_feature_correlations.csv")
    print("Saved plot: data/processed/plots/chunk_feature_correlations.png")


if __name__ == "__main__":
    main()
