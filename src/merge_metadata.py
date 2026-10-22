
from pathlib import Path
import csv

PROCESSED_DIR = Path("data/processed")
FEATURES_FILE = PROCESSED_DIR / "features.csv"
METADATA_FILE = PROCESSED_DIR / "story_metadata.csv"
OUTPUT_FILE = PROCESSED_DIR / "research_dataset.csv"


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def main():
    features = read_csv(FEATURES_FILE)
    metadata = read_csv(METADATA_FILE)

    metadata_by_filename = {
        row["filename"]: row for row in metadata
    }

    merged_rows = []

    for feature_row in features:
        filename = feature_row["filename"]

        if filename not in metadata_by_filename:
            raise ValueError(f"Missing metadata for {filename}")

        metadata_row = metadata_by_filename[filename]

        merged_rows.append({
            **metadata_row,
            **{
                key: value
                for key, value in feature_row.items()
                if key != "filename"
            },
        })

    if len(merged_rows) != len(metadata):
        raise ValueError("Feature and metadata row counts differ.")

    fieldnames = list(merged_rows[0].keys())

    with OUTPUT_FILE.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)

    print(f"Merged {len(merged_rows)} stories.")
    print(f"Saved dataset to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
