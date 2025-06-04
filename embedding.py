import json
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

def run_embedding_generation(input_dir: Path, output_dir: Path, model_name: str, device: torch.device):
    """
    Generates embeddings for each text chunk and saves them into new JSON files
    in the output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the embedding model
    print(f"Loading SentenceTransformer model: {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device)

    for in_path in input_dir.glob("*.json"):
        if in_path.name == "all_chunks_index.json":
            print(f"Skipping {in_path.name} as it's an index file.")
            continue

        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        text = data.get("text", "")
        if not text:
            print(f"Warning: Skipping {in_path.name} due to missing 'text' field.")
            continue
    
        embedding = model.encode(text, show_progress_bar=False).tolist()
        data["embedding"] = embedding

        # Save the updated data (with embedding) to the output directory
        out_path = output_dir / in_path.name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\\u2714\\ufe0f  {in_path.name} \\u2192 {out_path.name}") # Using unicode checkmark and arrow

    print("All embedding generation done.")

if __name__ == "__main__":
    from config import CHUNKS_DIR, EMBEDDED_CHUNKS_DIR, EMBEDDING_MODEL_NAME, DEVICE
    print("Running embedding generation module directly (for testing)...")
    run_embedding_generation(CHUNKS_DIR, EMBEDDED_CHUNKS_DIR, EMBEDDING_MODEL_NAME, DEVICE)