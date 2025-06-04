import json
from pathlib import Path

try:
    import nltk
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split `text` into word-based chunks of size `chunk_size` with `chunk_overlap`.
    Returns a list of chunk strings.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks

def run_chunking(input_txt_dir: Path, output_chunks_dir: Path, chunk_size: int, chunk_overlap: int):
    """
    Batch chunking of cleaned text files.
    Each chunk is saved as a separate JSON file with metadata.
    An index JSON file containing all chunk metadata and text is also saved.
    """
    output_chunks_dir.mkdir(parents=True, exist_ok=True)
    all_chunks_list = []

    for txt_file in sorted(input_txt_dir.glob("*.txt")):
        doc_id = txt_file.stem
        text = txt_file.read_text(encoding="utf-8")
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for idx, chunk in enumerate(chunks):
            metadata = {
                "doc_id": doc_id,
                "chunk_id": idx
            }
            record = {
                "text": chunk,
                "metadata": metadata
            }
            out_path = output_chunks_dir / f"{doc_id}_chunk{idx}.json"
            out_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            all_chunks_list.append(record)

    # Save an index of all chunks for easy loading later
    with open(output_chunks_dir / "all_chunks_index.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks_list, f, ensure_ascii=False, indent=2)
    print(f"Created {len(all_chunks_list)} chunks in {output_chunks_dir}")

if __name__ == "__main__":
    # This block runs only if the script is executed directly
    from config import PROCESSED_DOCS_DIR, CHUNKS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
    print("Running chunking module directly (for testing)...")
    run_chunking(PROCESSED_DOCS_DIR, CHUNKS_DIR, CHUNK_SIZE, CHUNK_OVERLAP)