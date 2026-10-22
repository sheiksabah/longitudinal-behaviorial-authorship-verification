from pathlib import Path
import re

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def extract_gutenberg_text(filename):
    path = RAW_DIR / filename
    text = path.read_text(encoding="utf-8", errors="replace")

    start_marker = re.search(
        r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK .*? \*\*\*",
        text,
        flags=re.IGNORECASE,
    )
    end_marker = re.search(
        r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK .*? \*\*\*",
        text,
        flags=re.IGNORECASE,
    )

    if not start_marker or not end_marker:
        raise ValueError(f"Could not find Gutenberg markers in {filename}")

    return text[start_marker.end():end_marker.start()].strip()


def main():
    stories = [
        "the_cask_of_amontillado.txt",
        "the_fall_of_the_house_of_usher.txt",
        "the_masque_of_the_red_death.txt",
    ]

    for filename in stories:
        clean_text = extract_gutenberg_text(filename)

        output_path = PROCESSED_DIR / filename
        output_path.write_text(clean_text, encoding="utf-8")

        word_count = len(re.findall(r"\b[\w']+\b", clean_text))
        print(f"{filename}: {word_count:,} words -> {output_path}")


if __name__ == "__main__":
    main()
