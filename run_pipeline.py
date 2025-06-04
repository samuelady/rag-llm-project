import os
from pathlib import Path

# Import functions from modules
from config import (
    DOCUMENTS_DIR, PROCESSED_DOCS_DIR, CHUNKS_DIR, EMBEDDED_CHUNKS_DIR,
    INDEXED_DATA_DIR, EMBEDDING_MODEL_NAME, LLM_MODEL_NAME, DEVICE,
    CHUNK_SIZE, CHUNK_OVERLAP
)
from preprocessing import run_preprocessing
from chunking import run_chunking
from embedding import run_embedding_generation
from indexing import run_indexing
from rag_system import run_rag_sequence

# Disable ChromaDB telemetry if you decide to install it and don't want telemetry
os.environ["CHROMA_DISABLE_TELEMETRY"] = "TRUE"

if __name__ == "__main__":
    print("Starting RAG Pipeline for Cybersecurity Documents...\n")

    # Ensure all necessary directories exist
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDED_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    INDEXED_DATA_DIR.mkdir(parents=True, exist_ok=True)


    # --- Step 1: Preprocessing and Cleaning ---
    print("--- Step 1: Preprocessing and Cleaning ---")
    run_preprocessing(DOCUMENTS_DIR, PROCESSED_DOCS_DIR)
    print("-" * 40 + "\n")

    # --- Step 2: Chunking ---
    print("--- Step 2: Chunking ---")
    run_chunking(PROCESSED_DOCS_DIR, CHUNKS_DIR, CHUNK_SIZE, CHUNK_OVERLAP)
    print("-" * 40 + "\n")

    # --- Step 3: Embedding Generation ---
    print("--- Step 3: Embedding Generation ---")
    print(f"Using device for embeddings: {DEVICE}")
    run_embedding_generation(CHUNKS_DIR, EMBEDDED_CHUNKS_DIR, EMBEDDING_MODEL_NAME, DEVICE)
    print("-" * 40 + "\n")

    # --- Step 4: Indexing ---
    print("--- Step 4: Indexing ---")
    index_output_path = INDEXED_DATA_DIR / "nn_index.pkl"
    metadata_output_path = INDEXED_DATA_DIR / "metadata_list.json"
    run_indexing(EMBEDDED_CHUNKS_DIR, index_output_path, metadata_output_path)
    print("-" * 40 + "\n")

    # --- Step 5 & 6: RAG Integration and Example Query ---
    print("--- Step 5 & 6: RAG Integration and Example Query ---")
    import joblib # Import joblib here as it's only needed for loading in this scope
    # Load NearestNeighbors index and metadata for RAG
    if not index_output_path.exists() or not metadata_output_path.exists():
        print("Error: Index or metadata files not found. RAG cannot proceed.")
    else:
        nn_index_loaded = joblib.load(index_output_path)
        with open(metadata_output_path, 'r', encoding='utf-8') as f:
            metadata_list_loaded = json.load(f)
        print("NearestNeighbors index and metadata loaded for RAG.")

        query = "What are best practices for zero trust in hospital IT?"
        print(f"Query: {query}")
        
        try:
            response = run_rag_sequence(
                query,
                nn_index_loaded,
                metadata_list_loaded,
                EMBEDDED_CHUNKS_DIR, # Pass the directory where embedded chunks are stored
                EMBEDDING_MODEL_NAME,
                LLM_MODEL_NAME,
                DEVICE
            )
            print("\n--- LLM Response ---")
            print(response)
        except NotImplementedError as e:
            print(f"RAG execution skipped: {e}")
        except Exception as e:
            print(f"An error occurred during RAG execution: {e}")
    print("-" * 40 + "\n")

    print("RAG Pipeline execution finished.")