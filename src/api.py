from fastapi import FastAPI
from pydantic import BaseModel
from src.embeddings import embed_query, embed_documents
from src.cache import SemanticCache
from src.vector_store import VectorStore
from src.clustering import FuzzyClusterer
from src.data_loader import load_data
import numpy as np

app = FastAPI(title="Semantic Search Cache API")
cache = SemanticCache()
vector_store = None
clusterer = None
is_initialized = False

def initialize():
    global vector_store, clusterer, is_initialized
    if is_initialized:
        return

    print("Loading documents...")
    docs = load_data()
    # Use a subset for faster startup
    docs = docs[:500]

    print(f"Generating embeddings for {len(docs)} documents...")
    embeddings = embed_documents(docs)

    print("Building vector store...")
    vector_store = VectorStore(dimension=embeddings.shape[1])
    vector_store.add_documents(embeddings, docs)

    print("Fitting fuzzy clusterer...")
    clusterer = FuzzyClusterer(n_clusters=5)
    clusterer.fit(embeddings)

    is_initialized = True
    print("Initialization complete!")

@app.on_event("startup")
def startup_event():
    initialize()

@app.get("/")
def root():
    return {"message": "Semantic Search Cache API", "docs": "/docs"}

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query_endpoint(request: QueryRequest):
    query_embedding = embed_query(request.query)

    # Check semantic cache first
    cached = cache.lookup(query_embedding)
    if cached:
        return {
            "query": request.query,
            **cached
        }

    # Perform vector search
    results = vector_store.search(query_embedding, k=5)

    # Get cluster distribution
    cluster_info = clusterer.cluster_distribution(query_embedding)

    # Store in cache
    cache.store(
        request.query,
        query_embedding,
        results,
        dominant_cluster=cluster_info["dominant_cluster"]
    )

    return {
        "query": request.query,
        "cache_hit": False,
        "result": results,
        "dominant_cluster": cluster_info["dominant_cluster"],
        "cluster_distribution": cluster_info["cluster_distribution"]
    }

@app.post("/search")
def search_endpoint(request: QueryRequest):
    """Vector similarity search without caching."""
    query_embedding = embed_query(request.query)
    results = vector_store.search(query_embedding, k=5)
    return {"query": request.query, "results": results}

@app.post("/clusters")
def cluster_endpoint(request: QueryRequest):
    """Get fuzzy cluster distribution for a query."""
    query_embedding = embed_query(request.query)
    cluster_info = clusterer.cluster_distribution(query_embedding)
    return {"query": request.query, **cluster_info}

@app.get("/cache/stats")
def cache_stats():
    return cache.stats()

@app.delete("/cache")
def clear_cache():
    cache.clear()
    return {"message": "cache cleared"}