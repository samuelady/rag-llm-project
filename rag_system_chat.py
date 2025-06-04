import json
import numpy as np
import joblib
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
import os

# Mock objects for LLM if it cannot be loaded (to allow retrieval part to run)
class MockTokenizer:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("LLM tokenizer is not loaded. Please ensure LLM setup is correct.")
    def decode(self, *args, **kwargs):
        raise NotImplementedError("LLM tokenizer is not loaded. Please ensure LLM setup is correct.")
    @property
    def eos_token_id(self):
        return 0 # A mock value, actual EOS token ID from a real tokenizer is needed for generation

class MockModel:
    def generate(self, *args, **kwargs):
        raise NotImplementedError("LLM model is not loaded. Please ensure LLM setup is correct.")
    def eval(self):
        pass # Mock eval method

# Global variables for models to avoid reloading
_embedding_model_for_queries = None
_llm_tokenizer = None
_llm_model = None
_llm_device = None

def load_query_embedding_model(model_name: str, device: torch.device):
    """Loads the SentenceTransformer model for generating query embeddings."""
    global _embedding_model_for_queries
    if _embedding_model_for_queries is None:
        print(f"Loading query embedding model: {model_name} on {device}")
        _embedding_model_for_queries = SentenceTransformer(model_name, device=device)
    return _embedding_model_for_queries

def embed_texts_for_query(texts: list[str], model_name: str, device: torch.device) -> np.ndarray:
    """Generates embeddings for a list of texts using the loaded query embedding model."""
    model = load_query_embedding_model(model_name, device)
    return model.encode(texts, show_progress_bar=False).tolist()

def load_llm_and_tokenizer(model_name: str, device: torch.device):
    """
    Loads the Large Language Model (LLM) and its tokenizer.
    Handles potential import errors for large model dependencies.
    """
    global _llm_tokenizer, _llm_model, _llm_device
    if _llm_tokenizer is None or _llm_model is None:
        print(f"Attempting to load LLM model and tokenizer: {model_name} on {device}")
        _llm_device = device

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            # Optional: for 8-bit/4-bit quantization, import BitsAndBytesConfig and configure
            # from transformers import BitsAndBytesConfig
            # bnb_config = BitsAndBytesConfig(...)

            _llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
            _llm_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16, # Or torch.bfloat16, adjust based on GPU support
                device_map="auto", # Recommended for large models, requires `accelerate`
                # quantization_config=bnb_config, # Uncomment if using quantization
            )
            _llm_model.eval() # Set model to evaluation mode for inference
            print(f"LLM '{model_name}' loaded successfully on {_llm_model.device}.")
        except ImportError as e:
            print(f"Error importing LLM libraries (e.g., transformers, accelerate, bitsandbytes): {e}")
            print("Falling back to mock LLM. Please install necessary libraries for full functionality.")
            _llm_tokenizer = MockTokenizer()
            _llm_model = MockModel()
        except Exception as e:
            print(f"An unexpected error occurred during LLM loading: {e}")
            print("Falling back to mock LLM. This might be due to insufficient VRAM or other setup issues.")
            _llm_tokenizer = MockTokenizer()
            _llm_model = MockModel()

    return _llm_tokenizer, _llm_model

def retrieve_chunks(query: str, nn_index, metadata_list: list, chunk_folder: Path, embedding_model_name: str, device: torch.device, top_k: int = 5) -> list[dict]:
    """
    Retrieves top_k most relevant chunks for a given query using a NearestNeighbors index.
    """
    query_embedding = embed_texts_for_query([query], embedding_model_name, device)

    # Ensure query_embedding is 2D, as kneighbors expects a 2D array
    query_embedding_np = np.array(query_embedding).reshape(1, -1)

    dists, idxs = nn_index.kneighbors(query_embedding_np, n_neighbors=top_k)
    results = []
    for dist, idx in zip(dists[0], idxs[0]):
        meta = metadata_list[idx]
        file_name = f"{meta['doc_id']}_chunk{meta['chunk_id']}.json"
        chunk_path = chunk_folder / file_name

        text = ""
        if chunk_path.exists():
            try:
                chunk_data = json.loads(chunk_path.read_text(encoding="utf-8"))
                text = chunk_data.get("text", "")
            except json.JSONDecodeError:
                print(f"Warning: Error decoding JSON from {file_name}. Skipping chunk text.")
                text = ""
        else:
            print(f"Warning: Chunk file not found: {file_name}. Skipping chunk text.")

        results.append({
            "doc_id": meta["doc_id"],
            "chunk_id": meta["chunk_id"],
            "text": text,
            "distance": float(dist)
        })
    return results

def build_rag_prompt(query: str, hits: list[dict]) -> str:
    """
    Constructs a prompt for the LLM using the original query and retrieved chunks.
    """
    prompt = (
        "You are an expert in cybersecurity. "
        "Use ONLY the following excerpts to answer, and cite sources in the format [Source: doc_id_chunk_id]. "
        "If the answer is not available in the provided context, state 'The information is not available in the provided documents.'\\n\\n"
    )
    for h in hits:
        source_id = f"{h['doc_id']}_chunk{h['chunk_id']}"
        prompt += f"--- Source: {source_id} ---\\n{h['text']}\\n\\n"
    prompt += f"Question: {query}\\nAnswer:"
    return prompt

def run_rag_sequence(query: str, nn_index, metadata_list: list, chunk_folder: Path,
                     embedding_model_name: str, llm_model_name: str, device: torch.device, top_k: int = 5) -> str:
    """
    Executes the RAG sequence: retrieves relevant chunks, builds a prompt,
    and generates a response using the LLM.
    """
    # Load LLM and tokenizer if not already loaded
    llm_tokenizer, llm_model = load_llm_and_tokenizer(llm_model_name, device)

    # Check if LLM is a mock object
    if isinstance(llm_model, MockModel):
        print("\\n--- LLM Not Loaded ---")
        print("Skipping RAG generation as the LLM model is not loaded. Please address LLM loading issues.")
        print("--------------------\\n")
        # Optionally, print the prompt that would have been sent
        hits = retrieve_chunks(query, nn_index, metadata_list, chunk_folder, embedding_model_name, device, top_k=top_k)
        prompt_preview = build_rag_prompt(query, hits)
        print("Prompt preview if LLM was loaded:\n", prompt_preview[:500], "...") # Print first 500 chars
        return "LLM not loaded. Cannot generate response."

    hits = retrieve_chunks(query, nn_index, metadata_list, chunk_folder, embedding_model_name, device, top_k=top_k)
    prompt = build_rag_prompt(query, hits)

    print("\\n--- Generated Prompt (for LLM) ---")
    print(prompt)
    print("----------------------------------\\n")

    inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

    with torch.no_grad(): # Disable gradient calculations for inference
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=256, # Limit new tokens to avoid very long responses
            temperature=0.7,    # Controls randomness (lower is more deterministic)
            top_p=0.9,          # Controls nucleus sampling (diversity)
            do_sample=True,     # Enable sampling
            pad_token_id=llm_tokenizer.eos_token_id,
            repetition_penalty=1.1, # Penalize repetition
            num_return_sequences=1
        )

    # Decode only the newly generated tokens, excluding the prompt
    response = llm_tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    # This block runs only if the script is executed directly
    from config import EMBEDDED_CHUNKS_DIR, INDEXED_DATA_DIR, EMBEDDING_MODEL_NAME, LLM_MODEL_NAME, DEVICE

    print("Running RAG System module directly (interactive mode)...")

    # Load NearestNeighbors index and metadata
    index_path = INDEXED_DATA_DIR / "nn_index.pkl"
    metadata_path = INDEXED_DATA_DIR / "metadata_list.json"

    if not index_path.exists() or not metadata_path.exists():
        print("Error: Index or metadata files not found. Please run indexing.py first.")
    else:
        nn_index_loaded = joblib.load(index_path)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata_list_loaded = json.load(f)
        print("NearestNeighbors index and metadata loaded for RAG.")

        # --- Interactive Query Input ---
        while True:
            query = input("\\nEnter your cybersecurity question (or 'quit' to exit):\\n> ")
            if query.lower() == 'quit':
                break

            print(f"Query: {query}")

            try:
                response = run_rag_sequence(
                    query,
                    nn_index_loaded,
                    metadata_list_loaded,
                    EMBEDDED_CHUNKS_DIR,
                    EMBEDDING_MODEL_NAME,
                    LLM_MODEL_NAME,
                    DEVICE
                )
                print("\n--- LLM Response ---")
                print(response)
            except NotImplementedError as e:
                print(f"RAG execution skipped: {e}")
                # Break the loop if LLM is not loaded, as further queries will also fail
                break
            except Exception as e:
                print(f"An error occurred during RAG execution: {e}")
                
    print("\nExiting RAG System.")