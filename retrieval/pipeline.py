import json
import uuid
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI  
from dotenv import load_dotenv
import os 
import streamlit as st

load_dotenv()
GEMINI_API_KEY= os.getenv('GOOGLE_API_KEY')
GEMINI_BASE_URL= "https://generativelanguage.googleapis.com/v1beta/openai/"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

DB_PATH = "data/vector_db"
COLLECTION_NAME = "hallucination_papers"
EMBEDDING_MODEL = "multi-qa-mpnet-base-dot-v1" 
LLM_CLIENT = OpenAI(base_url=OLLAMA_BASE_URL, api_key='OLLAMA') 

@st.cache_resource
def get_qdrant_client(path):
    """
    Creates a singleton connection to Qdrant. 
    Streamlit will maintain this single instance across reruns.
    """
    return QdrantClient(":memory:")
class RAGPipeline:
    def __init__(self):
        print("Loading embedding model...")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)
        self.reranker= CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.client = get_qdrant_client(DB_PATH)
        
        if not self.client.collection_exists(COLLECTION_NAME):
            self.index_knowledge_base()

    def index_knowledge_base(self):
        print("Indexing Chunks...")
        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.DOT)
        )
        
        with open("data/processed/knowledge_base_refined.jsonl", "r") as f:
            chunks = [json.loads(line) for line in f]
        texts_to_embed = [
            f"Paper: {c['metadata']['source_title']}. Section: {c['metadata']['section']}. {c['text']}" 
            for c in chunks
        ]
        vectors = self.encoder.encode(texts_to_embed, show_progress_bar=True)

        
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, c["chunk_id"])),
                vector=v.tolist(),
                payload=c
            ) for c, v in zip(chunks, vectors)
        ]
        
        self.client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"Indexed {len(points)} chunks.")
    def add_new_paper(self, chunks: List[dict]):
        """
        Dynamically encodes and upserts a list of chunks into Qdrant.
        Used for Streamlit uploads.
        """
        if not chunks:
            print("No chunks to add.")
            return

        print(f"Indexing {len(chunks)} new chunks...")
        
        # 1. Prepare strings for embedding
        texts_to_embed = [
            f"Paper: {c['metadata']['source_title']}. Section: {c['metadata']['section']}. {c['text']}" 
            for c in chunks
        ]
        
        # 2. Generate Vectors
        vectors = self.encoder.encode(texts_to_embed, show_progress_bar=True)

        # 3. Create Qdrant Points
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, c["chunk_id"])),
                vector=v.tolist(),
                payload=c
            ) for c, v in zip(chunks, vectors)
        ]
        
        # 4. Upsert (Qdrant handles 'upsert' as Update or Insert automatically)
        self.client.upsert(collection_name=COLLECTION_NAME, points=points)
        print("Successfully added new paper to knowledge base.")

    def retrieve(self, query: str, top_k=25):
        query_vector = self.encoder.encode(query).tolist()
        hits = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector
        )
        if not hits:
            return []
        
        cross_inp= [[query, hit.payload['text']] for hit in hits]
        cross_scores= self.reranker.predict(cross_inp)

        for idx,hit in enumerate(hits):
            hit.score= cross_scores[idx]

        hits = sorted(hits, key= lambda x:x.score, reverse=True)
        return [hit.payload for hit in hits[:top_k]]

if __name__ == "__main__":
    rag = RAGPipeline()