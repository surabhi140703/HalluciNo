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



