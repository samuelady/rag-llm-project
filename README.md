
# RAG-based LLM Evaluation on Cybersecurity & Pentesting Documents

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline to evaluate Large Language Models (LLMs) in the domain of **penetration testing and cybersecurity**. The goal is to compare LLM performance with and without contextual document retrieval.

---

## Project Overview

The pipeline performs the following:

1. Document chunking
2. Embedding chunks into vectors
3. Indexing vectors using FAISS
4. Querying LLMs with and without document context
5. Evaluation (Quantitative + Qualitative)

---

## Repository Structure

```
├── chunking.py                     # Split source documents into chunks
├── embedding.py                    # Convert chunks into embeddings
├── indexing.py                     # Index embeddings using FAISS
├── llm_with_rag_and_history.py     # Run LLM inference with RAG
├── llm_without_rag.py              # Run LLM inference without RAG
├── llm_evaluationv3.py             # Evaluate LLM responses
├── config.py                       # Configuration and constants
├── evaluation_data.parquet         # Test question set
├── llm_quantitative_eval.csv       # Evaluation metrics
├── llm_qualitative_eval.csv        # Sample answers with feedback
│
├── documents/                      # Raw input files (PDF/TXT)
├── documents_chunked/              # Chunked document JSONs
├── documents_embedded/             # Embeddings of chunks
├── documents_indexed/              # Indexed data (FAISS, metadata)
├── documents_test/                 # Input questions for evaluation
```

---

## Getting Started

### Installation

```bash
git clone <this-repo-url>
cd <repo-folder>

# Create environment
python -m venv venv
source venv/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

Example `requirements.txt`:

```
torch
transformers
pandas
scikit-learn
faiss-cpu
tqdm
matplotlib
sentence-transformers
```

---

## Execution Steps

### 1. Chunk Documents

```bash
python chunking.py
```

- **Input**: `documents/`
- **Output**: `documents_chunked/`

---

### 2. Generate Embeddings

```bash
python embedding.py
```

- **Input**: `documents_chunked/`
- **Output**: `documents_embedded/`

---

### 3. Index Embeddings

```bash
python indexing.py
```

- **Input**: `documents_embedded/`
- **Output**: `documents_indexed/nn_index.pkl`, `metadata_list.json`

---

### 4. Run LLM Inference

#### ➕ With RAG:

```bash
python llm_with_rag_and_history.py
```

#### ➖ Without RAG:

```bash
python llm_without_rag.py
```

---

### 5. Evaluate Model Responses

```bash
python llm_evaluationv3.py
```

- **Outputs**:
  - `llm_quantitative_eval.csv`
  - `llm_qualitative_eval.csv`

---

## Evaluation Metrics

- Precision, Recall, F1-Score
- BLEU / ROUGE / METEOR (optional)
- Manual grading for qualitative evaluation

---

