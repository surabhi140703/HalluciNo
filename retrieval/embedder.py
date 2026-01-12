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

