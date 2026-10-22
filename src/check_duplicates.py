from pathlib import Path
import re

RAW_DIR = Path("data/raw")
CORPUS_DIR = Path("data/processed/corpus")

STORIES = {
    "the_cask_of_amontillado.txt": [
        "the_works_of_edgar_allan_poe_volume_2.txt",
    ],
    "the_fall_of_the_house_of_usher.txt": [
        "the_works_of_edgar_allan_poe _volume_1.txt",
        "the_works_of_edgar_allan_poe_volume_2.txt",
    ],
    "the_masque_of_the_red_death.txt": [
        "the_works_of_edgar_allan_poe_volume_2.txt",
    ],
}


def normalize(text):
    text = text.casefold()
    return re.sub(r"\s+", " ", text).strip()


def extract_gutenberg_narrative(text):
    start = re.search(
        r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        text,
        re.IGNORECASE,
    )

    end = re.search(
        r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        text,
        re.IGNORECASE,
    )

    if not start or not end or end.start() <= start.end():
        raise ValueError("Could not identify Gutenberg story boundaries.")

    text = text[start.end():end.start()]

    # Remove the title and author heading before comparing.
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)

    while lines and (
        lines[0].strip().casefold().startswith("the ")
        or lines[0].strip().casefold().startswith("by ")
    ):
        lines.pop(0)

    return normalize("\n".join(lines))


def main():
    for story_file, volume_files in STORIES.items():
        standalone = extract_gutenberg_narrative(
            (RAW_DIR / story_file).read_text(
                encoding="utf-8", errors="replace"
            )
        )

        # Use a distinctive 500-character passage from the narrative.
        excerpt = standalone[len(standalone) // 3:
                             len(standalone) // 3 + 500]

        print(f"\nStory: {story_file}")

        for volume_file in volume_files:
            volume = normalize(
                (RAW_DIR / volume_file).read_text(
                    encoding="utf-8", errors="replace"
                )
            )

            print(f"  {volume_file}: excerpt match = {excerpt in volume}")


if __name__ == "__main__":
    main()
