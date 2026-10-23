
from pathlib import Path
import re

RAW_DIR = Path("data/raw")

# The three stories (separate)
STORY_TITLES = [
    "The Cask of Amontillado",
    "The Fall of the House of Usher",
    "The Masque of the Red Death",
]


def read_text(path):
    """Read a text file, replacing any invalid characters."""
    return path.read_text(encoding="utf-8", errors="replace")


def count_words(text):
    """Count words using a simple whitespace/punctuation-aware pattern."""
    return len(re.findall(r"\b[\w']+\b", text))


def normalize(text):
    """Normalize text for easier title searching."""
    return re.sub(r"\s+", " ", text).casefold()


def main():
    files = sorted(RAW_DIR.glob("*.txt"))

    if not files:
        print(f"No .txt files found in {RAW_DIR.resolve()}")
        return

    print("CORPUS INVENTORY")
    print("=" * 60)

    contents = {}

    for path in files:
        text = read_text(path)
        contents[path.name] = text

        print(f"\nFile: {path.name}")
        print(f"Characters: {len(text):,}")
        print(f"Words: {count_words(text):,}")

    print("\n\nSTORY TITLE SEARCH")
    print("=" * 60)

    for title in STORY_TITLES:
        print(f"\n{title}")

        for filename, text in contents.items():
            if "volume" in filename.casefold():
                found = normalize(title) in normalize(text)

                print(f"  {filename}: {'FOUND' if found else 'NOT FOUND'}")


if __name__ == "__main__":
    main()
