import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any
class SemanticCache:
    """
    Cache system that detects semantically similar queries
    using embedding similarity.
    """
    def __init__(self, similarity_threshold: float = 0.9):
        self.cache = []
        self.threshold = similarity_threshold
        self.hit_count = 0
        self.miss_count = 0
    def lookup(self, query_embedding: np.ndarray):
        for entry in self.cache:
            similarity = cosine_similarity(
                [query_embedding],
                [entry["embedding"]]
            )[0][0]
            if similarity >= self.threshold:
                self.hit_count += 1
                return {
                    "cache_hit": True,
                    "matched_query": entry["query"],
                    "similarity_score": float(similarity),
                    "result": entry["result"],
                    "dominant_cluster": entry["dominant_cluster"]
                }
        self.miss_count += 1
        return None
    def store(self, query: str, embedding, result, dominant_cluster):
        self.cache.append({
            "query": query,
            "embedding": embedding,
            "result": result,
            "dominant_cluster": dominant_cluster
        })
    def stats(self):
        total = len(self.cache)
        hit_rate = self.hit_count / \
            (self.hit_count + self.miss_count) \
            if (self.hit_count + self.miss_count) else 0
        return {
            "total_entries": total,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate
        }
    def clear(self):
        self.cache = []
        self.hit_count = 0
        self.miss_count = 0