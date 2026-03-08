import faiss
import numpy as np
from typing import List
class VectorStore:
    """
    Lightweight vector database built using FAISS.
    Stores document embeddings and supports fast similarity search.
    """
    def __init__(self, dimension: int):
        self.dimension = dimension
        # L2 similarity index
        self.index = faiss.IndexFlatL2(dimension)
        # store original documents
        self.documents: List[str] = []
    def add_documents(self, embeddings: np.ndarray, docs: List[str]) -> None:
        """
        Add documents and their embeddings to the vector store.
        """
        if len(embeddings) != len(docs):
            raise ValueError("Embeddings and documents count mismatch")
        self.index.add(embeddings.astype("float32"))
        self.documents.extend(docs)
    def search(self, query_embedding: np.ndarray, k: int = 5):
        """
        Perform similarity search.
        Returns top k most similar documents.
        """
        query_embedding = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_embedding, k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                results.append({
                    "document": self.documents[idx],
                    "distance": float(dist)
                })
        return results