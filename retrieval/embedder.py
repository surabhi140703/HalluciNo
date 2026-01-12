import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def build_retriever(jsonl_path, model_name="all-MiniLM-L6-v2"):
    print("Loading chunks")
    chunks=[]
    with open(jsonl_path, 'r', encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    texts= [c['text'] for c in chunks]
    print("Encoding chunks")
    model= SentenceTransformer(model_name)
    embeddings= model.encode(texts, show_progress_bar=True)

    #MiniLM dimension is 384
    dimension= embeddings.shape[1]
    faiss_index= faiss.IndexFlatIP(dimension)

    faiss.normalize_L2(embeddings)
    faiss_index.add(embeddings)

    faiss.write_index(faiss_index, "papers_index.faiss")
    np.save("embeddings.npy", embeddings)

    print("Index and embeddings saved")
    return model, faiss_index, chunks

def query_index(query, model, index, chunks, k=3):
    query_vector= model.encode([query])
    faiss.normalize_L2(query_vector)

    distances, indices= index.search(query_vector, k)
    for i in range(k):
        idx= indices[0][i]
        result= chunks[idx]
        print(f"Score: {distances[0][i]:.4f}")
        print(f"Source: {result['source']} (Page {result['page']})")
        print(f"Section: {result['section']}")
        print(f"Snippet: {result['text'][:200]}...")
        print("-" * 30)

if __name__ == "__main__":
    model, index, chunks = build_retriever("processed_data.jsonl")

    test_queries = [
        "What causes hallucination in multi-document summarisation?",
        "What evaluation metrics are used to measure hallucination?",
        "What is the definition of extrinsic hallucination?"
    ]
    for q in test_queries:
        query_index(q, model, index, chunks)    