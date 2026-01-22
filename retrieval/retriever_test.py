import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def load_assets(
    jsonl_path="processed_data.jsonl", 
    index_path="papers_index.faiss", 
    model_name="all-MiniLM-L6-v2"
):
    print(f"Loading metadata from {jsonl_path}")
    chunks = []
    with open(jsonl_path, 'r', encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    print(f"Loading FAISS index from {index_path}")
    index = faiss.read_index(index_path)

    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    return model, index, chunks

def retrieve(query, model, index, chunks, k=5):
    query_vec = model.encode([query])
    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec, k)

    results = []
    for i in range(k):
        idx = indices[0][i]
        results.append({
            "score": float(scores[0][i]),
            "source": chunks[idx]["source"],
            "section": chunks[idx]["section"],
            "page": chunks[idx]["page"], # Added this back so print_retrieved_chunks works
            "text": chunks[idx]["text"]
        })
    return results

def print_retrieved_chunks(query, results):
    print(f"\n" + "█"*80)
    print(f"QUERY: {query}")
    print("█"*80)
    
    for i, res in enumerate(results):
        print(f"\n[RANK {i+1}] | SCORE: {res['score']:.4f}")
        print(f"SOURCE: {res['source']} | SECTION: {res['section']} | PAGE: {res['page']}")
        print("-" * 40)
        print(f"{res['text']}")
        print("-" * 80)

# ---------- Execution ----------
if __name__ == "__main__":
    # 1. Load everything
    model, index, chunks = load_assets()

    # 2. Load the gold questions
    # Note: Ensure the path is correct (your previous prompt mentioned .jsonl but used .json logic)
    questions_path = "retrieval/retrieval_questions.json"
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"\nStarting Full Retrieval Inspection for {len(questions)} questions...")

    # 3. Loop through every question and print chunks
    hits = 0
    k_value = 3 # How many chunks to show per question

    for q in questions:
        query_text = q["question"]
        gold_sources = q["gold_sources"]
        
        # Perform retrieval
        retrieved_results = retrieve(query_text, model, index, chunks, k=k_value)
        
        # Check for HIT/MISS
        retrieved_filenames = [r["source"] for r in retrieved_results]
        is_hit = any(src in retrieved_filenames for src in gold_sources)
        hits += int(is_hit)
        
        # Print the detailed chunk info
        print_retrieved_chunks(query_text, retrieved_results)
        
        # Status footer for this question
        status = "✅ HIT" if is_hit else "❌ MISS"
        print(f"STATUS: {status}")
        print(f"EXPECTED SOURCES: {gold_sources}")
        print(f"FOUND SOURCES: {retrieved_filenames}")
        print("\n" + "═"*80 + "\n")

    # 4. Final Recall Score
    recall = hits / len(questions)
    print(f"FINAL RECALL@{k_value}: {recall:.2f}")