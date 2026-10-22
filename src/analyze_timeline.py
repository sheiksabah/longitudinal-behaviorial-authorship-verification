from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = Path("data/processed/research_dataset.csv")
OUTPUT_DIR = Path("data/processed/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(DATA_FILE)

    df = df.sort_values("first_publication_year")

    plt.figure(figsize=(10, 6))
    plt.scatter(
        df["first_publication_year"],
        df["average_sentence_length"],
    )

    plt.xlabel("First Publication Year")
    plt.ylabel("Average Sentence Length (words)")
    plt.title("Poe's Average Sentence Length by Publication Year")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output = OUTPUT_DIR / "sentence_length_timeline.png"
    plt.savefig(output, dpi=150)
    plt.close()

    print(f"Saved plot to {output}")
    print(df[[
        "title",
        "first_publication_year",
        "average_sentence_length",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
