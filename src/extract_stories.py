
from pathlib import Path
import re
import unicodedata

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed/extracted_volume2")

VOLUME = RAW_DIR / "the_works_of_edgar_allan_poe_volume_2.txt"

TITLES = [
    "The Purloined Letter",
    "The Thousand-and-Second Tale of Scheherazade",
    "A Descent into the Maelström",
    "Von Kempelen and His Discovery",
    "Mesmeric Revelation",
    "The Facts in the Case of M. Valdemar",
    "The Black Cat",
    "The Fall of the House of Usher",
    "Silence: A Fable",
    "The Masque of the Red Death",
    "The Cask of Amontillado",
    "The Imp of the Perverse",
    "The Island of the Fay",
    "The Assignation",
    "The Pit and the Pendulum",
    "The Premature Burial",
    "The Domain of Arnheim",
    "Landor's Cottage",
    "William Wilson",
    "The Tell-Tale Heart",
    "Berenice",
    "Eleonora",
]


def normalize_heading(value):
    """Normalize capitalization, accents, and punctuation."""
    value = unicodedata.normalize("NFKD", value)
    value = "".join(
        char for char in value
        if not unicodedata.combining(char)
    )

    value = value.casefold()
    value = value.replace("’", "'").replace("—", " ")
    value = re.sub(r"[^a-z0-9]+", " ", value)

    return " ".join(value.split())


def slugify(title):
    """Convert a story title into a filename."""
    return normalize_heading(title).replace(" ", "_") + ".txt"


def find_heading(text, title):
    """
    Find a standalone story heading.

    Returns the position immediately after the heading.
    The last match is used to avoid matching the table of contents.
    """
    target = normalize_heading(title)
    last_match = None

    for match in re.finditer(r"(?m)^[ \t]*(.+?)[ \t]*$", text):
        if normalize_heading(match.group(1)) == target:
            last_match = match

    if last_match is None:
        return None

    return last_match.end()


def count_words(text):
    """Count approximate words in a text."""
    return len(re.findall(r"\b[\w'-]+\b", text))


def remove_gutenberg_footer(text):
    """Remove the volume's notes section and Project Gutenberg footer."""
    patterns = [
        r"(?im)^\s*NOTES TO THE SECOND VOLUME\s*$",
        r"(?im)^\s*\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*",
        r"(?im)^\s*\*\*\*\s*END OF PROJECT GUTENBERG.*",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            text = text[:match.start()]

    return text.strip()


def main():
    if not VOLUME.exists():
        raise FileNotFoundError(f"Volume file not found: {VOLUME}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    text = VOLUME.read_text(
        encoding="utf-8",
        errors="replace"
    )

    headings = []

    for title in TITLES:
        position = find_heading(text, title)

        if position is None:
            print(f"NOT FOUND: {title}")
            continue

        headings.append((position, title))

    headings.sort(key=lambda item: item[0])

    print(f"\nFound {len(headings)} candidate headings.\n")

    for index, (start, title) in enumerate(headings):
        if index + 1 < len(headings):
            next_heading_position, _ = headings[index + 1]

            # Exclude the next heading line from the current story.
            end = text.rfind("\n", 0, next_heading_position)

            if end == -1:
                end = next_heading_position
        else:
            end = len(text)

        story_text = text[start:end].strip()
        story_text = remove_gutenberg_footer(story_text)

        word_count = count_words(story_text)

        if word_count < 100:
            print(f"SKIPPED: {title} ({word_count} words)")
            continue

        output_path = OUTPUT_DIR / slugify(title)

        output_path.write_text(
            story_text + "\n",
            encoding="utf-8"
        )

        print(
            f"Extracted: {title} "
            f"({word_count} words) -> {output_path}"
        )

    print("\nExtraction finished.")
    print("Inspect story boundaries before feature extraction.")


if __name__ == "__main__":
    main()
