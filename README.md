# Semantic Search Cache

This project implements a semantic search system with a caching layer to avoid repeating expensive computations for similar queries. It uses FAISS for vector similarity search, sentence-transformers for generating embeddings, and Gaussian Mixture Models for clustering queries. The system is exposed through a REST API built with FastAPI.

The goal of the project is to demonstrate how semantic search, clustering, and caching can work together to improve search efficiency.

---

## Features

- **Semantic Search**
  Documents are converted into vector embeddings and stored in a FAISS index, allowing fast similarity searches.

- **Sentence-Transformer Embeddings**
  The project uses the `all-MiniLM-L6-v2` model from sentence-transformers to generate dense vector representations of text.

- **Semantic Caching**
  Before performing a new search, the system checks whether a similar query has already been processed. If the similarity score is high enough, the cached result is returned.

- **Fuzzy Clustering**
  Queries are grouped using a Gaussian Mixture Model so that each query can belong to multiple clusters with different probabilities.

- **REST API**
  All functionality is accessible through a FastAPI application with interactive Swagger documentation.

---

## Project Structure

```
semantic-search-cache/
├ README.md
├ requirements.txt
├ data/
└ src/
    ├ __init__.py
    ├ api.py            # FastAPI application and API routes
    ├ cache.py          # Semantic cache implementation
    ├ clustering.py     # Gaussian Mixture Model clustering
    ├ data_loader.py    # Loads the 20 Newsgroups dataset
    ├ embeddings.py     # Generates sentence embeddings
    ├ search.py         # Main search pipeline
    └ vector_store.py   # FAISS index management
```

---

## Tech Stack

| Component     | Technology                              |
|---------------|-----------------------------------------|
| Embeddings    | sentence-transformers (all-MiniLM-L6-v2)|
| Vector Search | FAISS (faiss-cpu)                       |
| Clustering    | Gaussian Mixture Model (scikit-learn)   |
| Similarity    | Cosine Similarity                       |
| Dataset       | 20 Newsgroups                           |
| API           | FastAPI + Uvicorn                       |

---

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd semantic-search-cache
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

**Windows:**

```powershell
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the server

```bash
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

When the server starts, it will:

1. Load 500 documents from the 20 Newsgroups dataset
2. Generate embeddings for each document
3. Build the FAISS index
4. Train the Gaussian Mixture Model used for clustering

After initialization, the API will be available at:

> http://127.0.0.1:8000

---

## API Endpoints

### `GET /`

Basic health check endpoint.

### `POST /query`

Runs the complete pipeline: semantic search, clustering, and cache lookup.

**Example request:**

```json
{
  "query": "What is machine learning?"
}
```

**Example response (cache miss):**

```json
{
  "query": "What is machine learning?",
  "cache_hit": false,
  "result": [{"document": "...", "distance": 1.18}],
  "dominant_cluster": 2,
  "cluster_distribution": [0.0, 0.0, 1.0, 0.0, 0.0]
}
```

**Example response (cache hit):**

```json
{
  "query": "What is machine learning",
  "cache_hit": true,
  "matched_query": "What is machine learning?",
  "similarity_score": 0.958,
  "result": [{"document": "...", "distance": 1.18}],
  "dominant_cluster": 2
}
```

### `POST /search`

Runs vector similarity search without using the cache.

**Example:**

```json
{
  "query": "space exploration NASA"
}
```

### `POST /clusters`

Returns the cluster probability distribution for a query.

**Example:**

```json
{
  "query": "computer hardware"
}
```

### `GET /cache/stats`

Returns cache statistics including total entries, number of hits and misses, and hit rate.

### `DELETE /cache`

Clears all cached entries and resets cache statistics.

---

## System Overview

- **Dataset Loading**
  The project loads 500 documents from the 20 Newsgroups dataset using scikit-learn.

- **Embedding Generation**
  Each document is converted into a 384-dimensional embedding using the `all-MiniLM-L6-v2` model.

- **Vector Indexing**
  The embeddings are stored in a FAISS `IndexFlatL2` index to support fast similarity search.

- **Clustering**
  A Gaussian Mixture Model is trained on the embeddings to provide soft cluster assignments.

- **Query Processing**
  Incoming queries are embedded and compared with cached queries using cosine similarity. If no match is found, the FAISS index is searched.

- **Caching**
  Results are stored along with the query embedding so that future similar queries can reuse the cached result.

---

## Interactive API Documentation

Once the server is running, open:

> http://127.0.0.1:8000/docs

This page provides Swagger UI where you can test all API endpoints directly from the browser.