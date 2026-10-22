
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = Path("data/processed/research_dataset.csv")
OUTPUT_DIR = Path("data/processed/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(DATA_FILE)

    features = [
        "average_sentence_length",
        "sentence_length_std",
        "type_token_ratio",
        "comma_frequency_per_1000_words",
        "semicolon_frequency_per_1000_words",
    ]

    # Calculate each feature's correlation with publication year.
    correlations = df[features].corrwith(
        df["first_publication_year"]
    ).sort_values()

    print("Correlation with publication year:")
    print(correlations.round(3).to_string())

    # Plot the correlations.
    plt.figure(figsize=(10, 6))
    correlations.plot(kind="barh")
    plt.axvline(0, linewidth=1)
    plt.xlabel("Pearson correlation with publication year")
    plt.title("Writing Features and Publication Year")
    plt.tight_layout()

    output = OUTPUT_DIR / "temporal_feature_correlations.png"
    plt.savefig(output, dpi=150)
    plt.close()

    print(f"\nSaved plot to {output}")


if __name__ == "__main__":
    main()
