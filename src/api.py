from fastapi import FastAPI
from pydantic import BaseModel
from embeddings import embed_query
from semantic_cache import SemanticCache
app = FastAPI(title="Semantic Search Cache API")
cache = SemanticCache()
class QueryRequest(BaseModel):
    query: str
@app.post("/query")
def query_endpoint(request: QueryRequest):
    query_embedding = embed_query(request.query)
    cached = cache.lookup(query_embedding)
    if cached:
        return {
            "query": request.query,
            **cached
        }
    # placeholder result 
    result = "computed semantic search result"
    cache.store(
        request.query,
        query_embedding,
        result,
        dominant_cluster=0
    )
    return {
        "query": request.query,
        "cache_hit": False,
        "result": result
    }
@app.get("/cache/stats")
def cache_stats():
    return cache.stats()
@app.delete("/cache")
def clear_cache():
    cache.clear()
    return {"message": "cache cleared"}