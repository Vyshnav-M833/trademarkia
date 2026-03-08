from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
def embed_documents(docs):
    """
    Generate embeddings for a list of documents.
    """
    return model.encode(docs, convert_to_numpy=True)
def embed_query(query):
    """
    Generate embedding for a single query.
    """
    return model.encode([query], convert_to_numpy=True)[0]