import json
import numpy as np
import joblib
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
import os

# Mock objects for LLM if it cannot be loaded
class MockTokenizer:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("LLM tokenizer is not loaded. Please ensure LLM setup is correct.")
    def decode(self, *args, **kwargs):
        raise NotImplementedError("LLM tokenizer is not loaded. Please ensure LLM setup is correct.")
    @property
    def eos_token_id(self):
        return 0

class MockModel:
    def generate(self, *args, **kwargs):
        raise NotImplementedError("LLM model is not loaded. Please ensure LLM setup is correct.")
    def eval(self):
        pass

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

def load_llm_and_tokenizer_rag(model_name: str, device: torch.device):
    """
    Loads the Large Language Model (LLM) and its tokenizer for RAG usage.
    Handles potential import errors for large model dependencies.
    """
    global _llm_tokenizer, _llm_model, _llm_device
    if _llm_tokenizer is None or _llm_model is None:
        print(f"Attempting to load LLM model and tokenizer: {model_name} on {device}")
        _llm_device = device

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            _llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
            _llm_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16, # Adjust based on GPU support
                device_map="auto",         # Recommended for large models, requires `accelerate`
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

def build_rag_conversation_prompt(question: str, chat_history: list[tuple[str, str]], hits: list[dict]):
    """
    Builds a prompt for the LLM including chat history and RAG context.
    `chat_history` is a list of (user_message, llm_response) tuples.
    """
    system_prompt = (
        "You are an expert in cybersecurity. Answer the following question concisely, "
        "using ONLY the provided excerpts as context. Cite your sources in the format [Source: doc_id_chunk_id] "
        "at the end of each relevant sentence or phrase. Do NOT include redundant information. "
        "Be direct and avoid conversational filler. If the complete answer is not available "
        "in the provided documents, state 'The information is not fully available in the provided documents.' "
        "Do NOT introduce external information or make assumptions."
    )
    
    prompt_parts = [f"### System:\\n{system_prompt}\\n"]
    
    # Add retrieved context first, ensuring it's clearly marked as context
    prompt_parts.append("### Relevant Documents:\\n")
    for h in hits:
        source_id = f"{h['doc_id']}_chunk{h['chunk_id']}"
        prompt_parts.append(f"--- Source: {source_id} ---\\n{h['text']}\\n")
    prompt_parts.append("\\n") # Add a newline to separate context from history/current question
        
    # Add chat history
    for user_msg, llm_resp in chat_history:
        prompt_parts.append(f"### User:\\n{user_msg}\\n")
        prompt_parts.append(f"### Assistant:\\n{llm_resp}\\n")
        
    # Add current question
    prompt_parts.append(f"### User:\\n{question}\\n### Assistant:\\n")
    
    return "".join(prompt_parts)

def generate_response_with_rag(question: str, chat_history: list[tuple[str, str]], nn_index, metadata_list: list, chunk_folder: Path, embedding_model_name: str, llm_model_name: str, device: torch.device, top_k: int = 5):
    """
    Generates a response from the LLM with RAG and considering chat history.
    """
    llm_tokenizer, llm_model = load_llm_and_tokenizer_rag(llm_model_name, device)

    if isinstance(llm_model, MockModel):
        print("\\n--- LLM Not Loaded ---")
        print("Skipping generation as the LLM model is not loaded. Please address LLM loading issues.")
        print("--------------------\\n")
        return "LLM not loaded. Cannot generate response with RAG."

    # Retrieve relevant chunks based on the current question
    # You might also consider retrieving based on the combined question + last turn of chat history
    hits = retrieve_chunks(question, nn_index, metadata_list, chunk_folder, embedding_model_name, device, top_k=top_k)
    
    prompt = build_rag_conversation_prompt(question, chat_history, hits)

    print("\\n--- Generated Prompt (for LLM - with RAG) ---")
    print(prompt)
    print("----------------------------------------------\\n")

    inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=250, # Adjusted for conciseness with RAG
            temperature=0.3,    # Lower temperature for less creativity, more factual adherence
            top_p=0.9,
            do_sample=True,
            pad_token_id=llm_tokenizer.eos_token_id,
            repetition_penalty=1.1,
            num_return_sequences=1
        )

    response = llm_tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    return response.strip()

if __name__ == "__main__":
    from config import EMBEDDED_CHUNKS_DIR, INDEXED_DATA_DIR, EMBEDDING_MODEL_NAME, LLM_MODEL_NAME, DEVICE, get_optimal_device

    print("Running LLM With RAG (Interactive Chat Mode)...")
    # Ensure device is correctly set for the interactive session
    DEVICE = get_optimal_device()
    
    # Load NearestNeighbors index and metadata
    index_path = INDEXED_DATA_DIR / "nn_index.pkl"
    metadata_path = INDEXED_DATA_DIR / "metadata_list.json"
    
    if not index_path.exists() or not metadata_path.exists():
        print("Error: Index or metadata files not found. Please run indexing.py first via run_pipeline.py.")
        exit() # Exit if RAG components are missing
        
    nn_index_loaded = joblib.load(index_path)
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata_list_loaded = json.load(f)
    print("NearestNeighbors index and metadata loaded for RAG testing.")

    chat_history = [] # Stores (user_message, llm_response) tuples

    while True:
        user_question = input("\\nUser: ")
        if user_question.lower() == 'quit':
            break

        try:
            llm_response = generate_response_with_rag(
                user_question,
                chat_history,
                nn_index_loaded,
                metadata_list_loaded,
                EMBEDDED_CHUNKS_DIR,
                EMBEDDING_MODEL_NAME,
                LLM_MODEL_NAME,
                DEVICE
            )
            print("\\nAssistant (with RAG):", llm_response)
            chat_history.append((user_question, llm_response))
            
        except NotImplementedError as e:
            print(f"Error: {e}")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    print("\\nExiting LLM with RAG chat.")