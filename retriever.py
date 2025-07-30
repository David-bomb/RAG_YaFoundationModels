from typing import Dict, Any
from sentence_transformers import SentenceTransformer

def retrieve_chunks(
    query: str,
    top_k: int,
    embed_model: SentenceTransformer,
    collection
) -> Dict[str, Any]:
    q_emb = embed_model.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=['documents', 'metadatas', 'distances']
    )
    return results
