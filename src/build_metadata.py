from pathlib import Path
import csv
import re

CORPUS_DIR = Path("data/processed/corpus")
METADATA_FILE = Path("data/processed/story_metadata.csv")


def count_words(text):
    return len(re.findall(r"\b[\w'-]+\b", text))


def main():
    with METADATA_FILE.open(
        "r", encoding="utf-8", newline=""
    ) as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = reader.fieldnames

    for row in rows:
        path = CORPUS_DIR / row["filename"]
        text = path.read_text(encoding="utf-8")
        row["word_count"] = count_words(text)

    with METADATA_FILE.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated word counts for {len(rows)} stories.")


if __name__ == "__main__":
    main()
