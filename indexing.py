import json
import numpy as np
import joblib
from pathlib import Path
from sklearn.neighbors import NearestNeighbors # Imported here as it's used only in this module

def run_indexing(chunk_json_folder: Path, index_output_path: Path, metadata_output_path: Path):
    """
    Builds a NearestNeighbors index from JSON files containing embeddings
    and saves the index and associated metadata to specified paths.
    """
    # Ensure the output directory for indexed data exists
    index_output_path.parent.mkdir(parents=True, exist_ok=True)

    embeddings = []
    metadatas = []
    
    print(f"Loading embeddings and metadata from {chunk_json_folder}...")
    for f in sorted(chunk_json_folder.glob("*.json")):
        # Skip the aggregate index file if it exists and contains a list
        if f.name == "all_chunks_index.json":
            print(f"Skipping {f.name} as it's an index file.")
            continue
        
        try:
            with open(f, "r", encoding="utf-8") as file:
                record = json.load(file)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {f.name}. Skipping.")
            continue

        if "embedding" in record and "metadata" in record:
            embeddings.append(record["embedding"])
            metadatas.append(record["metadata"])
        else:
            print(f"Warning: '{f.name}' is missing 'embedding' or 'metadata'. Skipping.")

    if not embeddings:
        raise ValueError(f"No valid embeddings found in {chunk_json_folder}. Cannot build index.")

    embeddings_np = np.array(embeddings, dtype=np.float32)
    print(f"Loaded {len(embeddings_np)} embeddings.")

    # Build the NearestNeighbors index using cosine similarity
    print("Building NearestNeighbors index...")
    nn_index = NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='brute')
    nn_index.fit(embeddings_np)
    
    # Save the fitted index
    joblib.dump(nn_index, index_output_path)
    print(f"NearestNeighbors index saved to: {index_output_path}")

    # Save the metadata list
    with open(metadata_output_path, 'w', encoding='utf-8') as f:
        json.dump(metadatas, f, ensure_ascii=False, indent=2)
    print(f"Metadata list saved to: {metadata_output_path}")

    print("Indexing complete.")

if __name__ == "__main__":
    # This block runs only if the script is executed directly
    from config import EMBEDDED_CHUNKS_DIR, INDEXED_DATA_DIR
    print("Running indexing module directly (for testing)...")
    
    index_file = INDEXED_DATA_DIR / "nn_index.pkl"
    metadata_file = INDEXED_DATA_DIR / "metadata_list.json"
    
    run_indexing(EMBEDDED_CHUNKS_DIR, index_file, metadata_file)