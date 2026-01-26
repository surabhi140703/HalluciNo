import json
import uuid
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from openai import OpenAI  # Or your preferred LLM client

GEMINI_API_KEY="AIzaSyCr14AMI_tUwvujqAfo6pWvfyFBcHMqxJE"
DB_PATH = "data/vector_db"
COLLECTION_NAME = "hallucination_papers"
EMBEDDING_MODEL = "multi-qa-mpnet-base-dot-v1" 
LLM_CLIENT = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=GEMINI_API_KEY) # Ensure env var is set

class RAGPipeline:
    def __init__(self):
        print("Loading embedding model...")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)
        self.client = QdrantClient(path=DB_PATH)
        
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

    def retrieve(self, query: str, top_k=5):
        query_vector = self.encoder.encode(query).tolist()
        hits = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k
        )
        return [hit.payload for hit in hits]

    def reason_and_answer(self, query: str):
        print(f"Querying: {query}")
        evidence = self.retrieve(query)
        
        context_str = ""
        for i, chunk in enumerate(evidence):
            context_str += f"""
            [Document {i+1}]
            Source: {chunk['metadata']['source_title']}
            Section: {chunk['metadata']['section']}
            Type: {chunk['metadata']['type']}
            Content: {chunk['text']}
            -----------------------------------
            """

        #refusal aware prompting
        system_prompt = """
        You are a strict academic researcher. You answer questions based ONLY on the provided documents.
        
        Rules:
        1. Conflict Detection: Check if documents contradict each other (e.g. Paper A says X, Paper B says Not X). Explicitly mention this.
        2. Confidence Score: End your response with a JSON object: {"confidence": 0.0-1.0, "reason": "..."}.
        3. Give arxiv ids first before quoting any papers.
        4. Refusal: If the documents do not contain the answer, say "I cannot answer this based on the provided context" and set confidence to 0.0.
        5. Citations: Cite the source paper title for every claim. 
        """
        
        response = LLM_CLIENT.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"}
            ]
        )
        
        return response.choices[0].message.content

if __name__ == "__main__":
    rag = RAGPipeline()
    answer = rag.reason_and_answer("Is hallucination primarily a retrieval problem?")
    print("\nModel response: \n")
    print(answer)