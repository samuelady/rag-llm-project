import pandas as pd
import json
from pathlib import Path
import time
import torch
import numpy as np
from tqdm.auto import tqdm # For progress bar

# Import functions from your existing modules
from config import (
    LLM_MODEL_NAME, DEVICE, EMBEDDING_MODEL_NAME,
    INDEXED_DATA_DIR, EMBEDDED_CHUNKS_DIR
)
from llm_without_rag import generate_response_without_rag
from llm_with_rag_and_history import generate_response_with_rag as generate_response_with_rag_and_history # Renamed to avoid confusion
from rag_system import retrieve_chunks # Re-use the chunk retrieval logic

# For loading RAG components
import joblib

import warnings
warnings.filterwarnings(
    "ignore",
    message="To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).",
    category=UserWarning
)

# --- Evaluation Data Path ---
EVAL_PARQUET_PATH = Path("documents_test/train-00000-of-00001.parquet") # Your specified path

# --- Output Paths ---
QUANTITATIVE_EVAL_CSV = Path("llm_quantitative_eval.csv")
QUALITATIVE_EVAL_CSV = Path("llm_qualitative_eval.csv")

def load_rag_components(indexed_data_dir: Path):
    """Loads the NearestNeighbors index and metadata for RAG."""
    index_path = indexed_data_dir / "nn_index.pkl"
    metadata_path = indexed_data_dir / "metadata_list.json"

    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"RAG components not found: {index_path} or {metadata_path}. "
            "Please run `python run_pipeline.py` entirely first to generate these files."
        )

    nn_index = joblib.load(index_path)
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)
    print("RAG components (NearestNeighbors index, metadata) loaded successfully.")
    return nn_index, metadata_list

# --- Quantitative Metrics ---
def calculate_exact_match(predicted_answer: str, ground_truth: str) -> bool:
    """Calculates Exact Match (EM) score."""
    if not ground_truth or ground_truth.strip().lower() == 'n/a':
        return np.nan # Not applicable if no ground truth
    return predicted_answer.strip().lower() == ground_truth.strip().lower()

def run_evaluation(parquet_file_path: Path, quantitative_csv_path: Path, qualitative_csv_path: Path):
    """
    Runs the evaluation for LLM with and without RAG based on questions
    from a Parquet file. Saves quantitative and qualitative results to CSV files.
    """
    print(f"Starting LLM evaluation from {parquet_file_path}...")

    if not parquet_file_path.exists():
        print(f"Error: Parquet file not found at {parquet_file_path}")
        return

    # Create parent directories for output CSVs
    quantitative_csv_path.parent.mkdir(parents=True, exist_ok=True)
    qualitative_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Load evaluation dataset
    try:
        eval_df = pd.read_parquet(parquet_file_path)
        # --- IMPORTANT FOR TESTING ---
        # For quick testing, uncomment the line below to process only a few questions
        eval_df = eval_df.head(5) # REMOVE OR COMMENT THIS LINE FOR FULL EVALUATION
        # -----------------------------
        print(f"Loaded {len(eval_df)} questions from Parquet file.")
        if 'question' not in eval_df.columns:
            raise ValueError("Parquet file must contain a 'question' column.")
    except Exception as e:
        print(f"Error reading Parquet file: {e}")
        return

    # Load RAG components
    nn_index, metadata_list = None, None
    rag_components_loaded = False
    try:
        nn_index, metadata_list = load_rag_components(INDEXED_DATA_DIR)
        rag_components_loaded = True
    except FileNotFoundError as e:
        print(f"RAG components not found: {e}. RAG evaluation will be skipped.")
    except Exception as e:
        print(f"Error loading RAG components: {e}. RAG evaluation will be skipped.")

    # Initialize lists for results
    quantitative_results_list = []
    qualitative_results_list = []

    # Print LLM loading note once before the loop starts
    print("\n--- LLM Loading Note ---")
    print("The LLM (Llama 2 7B) is large and might take time/resources to load. It will load when first used.")
    print("If you encounter errors, please review config.py for hardware requirements and setup notes.")
    print("--------------------\\n")

    # Main evaluation loop
    for i, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Evaluating Questions"):
        question = row['question']
        ground_truth = row.get('ground_truth_answer', 'N/A')
        print(f"\nEvaluating Question {i+1}/{len(eval_df)}: '{question[:70]}...'") # Print start of question

        # --- Evaluate LLM without RAG ---
        llm_no_rag_response = "Error: LLM not loaded or unexpected error."
        duration_no_rag = 0
        print(f"  > Running LLM without RAG for Q{i+1}...")
        try:
            start_time = time.time()
            llm_no_rag_response = generate_response_without_rag(question, [], LLM_MODEL_NAME, DEVICE)
            duration_no_rag = time.time() - start_time
            print(f"  > No RAG response generated. Duration: {duration_no_rag:.2f}s")
        except NotImplementedError as e:
            llm_no_rag_response = f"LLM Not Loaded (No RAG): {e}"
            print(f"  > No RAG skipped: {e}")
        except Exception as e:
            llm_no_rag_response = f"Error in No RAG LLM: {e}"
            print(f"  > No RAG error: {e}")
        
        # --- Evaluate LLM with RAG ---
        llm_rag_response = "Error: LLM not loaded or RAG components missing."
        duration_rag = 0
        current_retrieved_hits = [] 
        retrieval_metrics = {} 
        
        print(f"  > Running LLM with RAG for Q{i+1}...")
        if rag_components_loaded:
            try:
                # Retrieve chunks first
                current_retrieved_hits = retrieve_chunks(question, nn_index, metadata_list, EMBEDDED_CHUNKS_DIR, EMBEDDING_MODEL_NAME, DEVICE)

                start_time = time.time()
                llm_rag_response = generate_response_with_rag_and_history( # Use the aliased function name
                    question, [], # chat_history, empty for single question evaluation
                    nn_index, metadata_list, EMBEDDED_CHUNKS_DIR,
                    EMBEDDING_MODEL_NAME, LLM_MODEL_NAME, DEVICE
                )
                duration_rag = time.time() - start_time
                print(f"  > RAG response generated. Duration: {duration_rag:.2f}s")
            except NotImplementedError as e:
                llm_rag_response = f"LLM Not Loaded (With RAG): {e}"
                print(f"  > RAG skipped: {e}")
            except Exception as e:
                llm_rag_response = f"Error in RAG LLM: {e}"
        else:
            llm_rag_response = "RAG evaluation skipped due to missing components."
            print(f"  > RAG skipped due to missing components.")

        # --- Quantitative Data Collection ---
        quantitative_results = {
            'question_id': i,
            'question': question,
            'ground_truth_answer': ground_truth,
            'llm_without_rag_response_len': len(llm_no_rag_response),
            'llm_with_rag_response_len': len(llm_rag_response),
            'duration_without_rag_sec': duration_no_rag,
            'duration_with_rag_sec': duration_rag,
            'em_without_rag': calculate_exact_match(llm_no_rag_response, ground_truth),
            'em_with_rag': calculate_exact_match(llm_rag_response, ground_truth),
        }
        quantitative_results_list.append(quantitative_results)

        # --- Qualitative Data Collection (for human review) ---
        qualitative_results = {
            'question_id': i,
            'question': question,
            'ground_truth_answer': ground_truth,
            'llm_without_rag_response': llm_no_rag_response,
            'llm_with_rag_response': llm_rag_response,
            # Information about retrieved chunks for qualitative assessment
            'retrieved_chunks_info': [
                {
                    'doc_id': hit['doc_id'],
                    'chunk_id': hit['chunk_id'],
                    'distance': hit['distance'],
                    'text_preview': hit['text'][:200] + '...' if len(hit['text']) > 200 else hit['text']
                } for hit in current_retrieved_hits
            ],
            # Human evaluation columns (to be filled manually)
            'factual_accuracy_no_rag': np.nan, 
            'relevance_no_rag': np.nan,      
            'completeness_no_rag': np.nan,   
            'conciseness_no_rag': np.nan,    
            'hallucination_no_rag': np.nan,  
            'factual_accuracy_with_rag': np.nan, 
            'relevance_with_rag': np.nan,      
            'completeness_with_rag': np.nan,   
            'conciseness_with_rag': np.nan,    
            'hallucination_with_rag': np.nan,  
            'preference_rag_vs_no_rag': np.nan 
        }
        qualitative_results_list.append(qualitative_results)

    # Save final results to CSV
    pd.DataFrame(quantitative_results_list).to_csv(quantitative_csv_path, index=False)
    print(f"\nQuantitative evaluation results saved to {quantitative_csv_path}")

    # For qualitative, convert list/dict columns to string representation for CSV compatibility
    qual_df = pd.DataFrame(qualitative_results_list)
    for col in ['retrieved_chunks_info']:
        if col in qual_df.columns:
            qual_df[col] = qual_df[col].apply(json.dumps, ensure_ascii=False)

    qual_df.to_csv(qualitative_csv_path, index=False)
    print(f"Qualitative evaluation template saved to {qualitative_csv_path}")

    print("\nEvaluation process complete. Please review the CSV files for detailed analysis and human annotation.")

if __name__ == "__main__":
    # Ensure the 'documents_test' directory exists relative to BASE_DIR
    from config import BASE_DIR
    test_data_abs_path = BASE_DIR / EVAL_PARQUET_PATH # Construct absolute path

    # Run the evaluation
    run_evaluation(test_data_abs_path, QUANTITATIVE_EVAL_CSV, QUALITATIVE_EVAL_CSV)