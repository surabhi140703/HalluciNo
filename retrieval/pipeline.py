import json
import uuid
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI  # Or your preferred LLM client
from dotenv import load_dotenv

GEMINI_API_KEY= os.getenv("GOOGLE_API_KEY")
DB_PATH = "data/vector_db"
COLLECTION_NAME = "hallucination_papers"
EMBEDDING_MODEL = "multi-qa-mpnet-base-dot-v1" 
LLM_CLIENT = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=GEMINI_API_KEY) # Ensure env var is set

class RAGPipeline:
    def __init__(self):
        print("Loading embedding model...")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)
        self.reranker= CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
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
            limit=25
        )
        if not hits:
            return []
        
        cross_inp= [[query, hit.payload['text']] for hit in hits]
        cross_scores= self.reranker.predict(cross_inp)

        for idx,hit in enumerate(hits):
            hit.score= cross_scores[idx]

        hits = sorted(hits, key= lambda x:x.score, reverse=True)
        return [hit.payload for hit in hits[:top_k]]

    def reason_and_answer(self, query: str):
        print(f"Querying: {query}")
        evidence = self.retrieve(query)
        
        context_str = ""
        for i, chunk in enumerate(evidence):
            context_str += f"""
            ### REFERENCE: {chunk['metadata']['source_title']}
            SECTION: {chunk['metadata']['section']}
            CONTENT: {chunk['text']}
            -----------------------------------
            """

        #refusal aware prompting
        system_prompt = """
        You are a senior academic researcher conducting a literature review. 
        Your goal is to synthesize findings from multiple papers, not just summarize them.

        MANDATORY REASONING PROCESS:
        1. Identify Perspectives: Look for different answers to the question in different documents. (e.g., "Paper A suggests X, while Paper B suggests Y").
        2. Detect Conflicts: Explicitly state if one paper challenges the assumptions of another.
        3. Trace Lineage: If a paper refutes a common baseline assumption, frame it as a correction to the field.
        4. Final Synthesis: Your answer must reflect the nuance of the debate.

        OUTPUT FORMAT:
        - {paper name, section} claims ... for each paper chunk retrieved
        - Direct Answer: A nuanced summary (e.g., "While early baselines assumed X, recent evidence suggests Y...").
        - Evidence: - [Source A] claims...
        - However, [Source B] demonstrates...
        - Confidence: {"confidence": 0.0-1.0}
        CITATION RULE:
        Do NOT refer to "Document 1" or "Source 2". 
        ALWAYS refer to the paper by its specific **TITLE** when making a claim. 
        (e.g., "As demonstrated in 'On the Role of Retrieval'...")
        If no solid evidence found, say "sorry out of syllabus" without quoting any citations
        """
        
        response = LLM_CLIENT.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"}
            ]
        )
        print(context_str)
        return response.choices[0].message.content

if __name__ == "__main__":
    rag = RAGPipeline()
    answer = rag.reason_and_answer("What did i eat today?")
    print("\nModel response: \n")
    print(answer)