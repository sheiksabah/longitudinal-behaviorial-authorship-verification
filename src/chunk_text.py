
from pathlib import Path
import re

# Directory configuration
INPUT_DIR = Path("data/processed/corpus")
OUTPUT_DIR = Path("data/processed/chunks")

# Number of words per chunk
CHUNK_SIZE = 1000


def chunk_text(text, chunk_size):
    """Split text into non-overlapping chunks while preserving punctuation."""

    word_matches = list(re.finditer(r"\b[\w']+\b", text))

    chunks = []

    for i in range(0, len(word_matches), chunk_size):
        start = word_matches[i].start()

        last_index = min(i + chunk_size, len(word_matches)) - 1
        end = word_matches[last_index].end()

        chunk = text[start:end]
        chunks.append(chunk)

    return chunks


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_chunks = 0

    for file_path in sorted(INPUT_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")

        chunks = chunk_text(text, CHUNK_SIZE)

        # Create a separate directory for each story
        story_name = file_path.stem
        story_output_dir = OUTPUT_DIR / story_name
        story_output_dir.mkdir(parents=True, exist_ok=True)

        # Save individual chunks
        for index, chunk in enumerate(chunks, start=1):
            output_file = story_output_dir / f"chunk_{index:03}.txt"
            output_file.write_text(chunk, encoding="utf-8")

        total_chunks += len(chunks)

        print(f"{file_path.name}: {len(chunks)} chunks created")

    print("\nChunking complete!")
    print(f"Total chunks created: {total_chunks}")


if __name__ == "__main__":
    main()
