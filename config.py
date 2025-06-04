from pathlib import Path
import torch

# ==============================================================================
# Global Configuration and Setup
# ==============================================================================

# --- File Paths Configuration ---
BASE_DIR = Path("/Users/samuelsanjaya/Documents/Project Cooperation /Pusan University/RAG Cybersecurity v5")

DOCUMENTS_DIR = BASE_DIR / "documents"
PROCESSED_DOCS_DIR = BASE_DIR / "documents_processed"
CHUNKS_DIR = BASE_DIR / "documents_chunked"
EMBEDDED_CHUNKS_DIR = BASE_DIR / "documents_embedded"
INDEXED_DATA_DIR = BASE_DIR / "documents_indexed"

# Ensure all necessary directories exist
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DOCS_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDED_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
INDEXED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# --- Model Configuration ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

# --- Chunking Parameters ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Device Configuration ---
def get_optimal_device():
    """Determines and returns the optimal device for Torch operations."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

DEVICE = get_optimal_device()

if not hasattr(torch, "get_default_device"):
    torch.get_default_device = lambda: DEVICE