import numpy as np

from embeddings import embed_query
from vector_store import VectorStore
from semantic_cache import SemanticCache
from clustering import FuzzyClusterer


class SemanticSearchEngine:

    def __init__(self, vector_store: VectorStore,
                 clusterer: FuzzyClusterer,
                 cache: SemanticCache):

        self.vector_store = vector_store
        self.clusterer = clusterer
        self.cache = cache

    def query(self, query_text: str):

        # Convert query to embedding
        query_embedding = embed_query(query_text)

        # Check semantic cache
        cached = self.cache.lookup(query_embedding)

        if cached:
            return cached

        # Perform vector search
        results = self.vector_store.search(query_embedding)

        # Determine cluster distribution
        cluster_info = self.clusterer.cluster_distribution(query_embedding)

        response = {
            "query": query_text,
            "cache_hit": False,
            "matched_query": None,
            "similarity_score": None,
            "result": results,
            "dominant_cluster": cluster_info["dominant_cluster"]
        }

        # Store in cache
        self.cache.store(
            query_text,
            query_embedding,
            results,
            cluster_info["dominant_cluster"]
        )

        return response