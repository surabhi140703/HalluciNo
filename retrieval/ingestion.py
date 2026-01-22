import os 
import json
import re
import arxiv
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions


ARXIV_IDS = [
    "2401.00396",  # RAGTruth: Hallucination corpus for RAG models
    "2303.08896",  # SelfCheckGPT: Black‑Box hallucination detection
    "2305.14251",  # FActScore: Atomic factual evaluation
    "2405.01563",  # Mitigating LLM Hallucinations via Conformal Abstention
    "2502.12964",  # Trust Me, I'm Wrong: High‑Certainty Hallucinations
    "2512.15068",  # The Semantic Illusion: Limits of embedding detection
]

DATA_DIR = Path("data/raw_pdfs")
OUTPUT_FILE = Path("data/processed/knowledge_base.jsonl")

def fetch_arxiv_pdfs(id_list):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    client= arxiv.Client()
    search= arxiv.Search(id_list= id_list)

    downloaded_paths=[]

    for paper in client.results(search):
        clean_title = re.sub(r'[^\w\-_\. ]', '_', paper.title)[:50]
        filename = f"{paper.get_short_id()}_{clean_title}.pdf"
        file_path= DATA_DIR/filename

        if not file_path.exists():
            print(f" downloading {paper.title}")
            paper.download_pdf(dirpath=DATA_DIR, filename=filename)
        else:
            print(f" cached {paper.title}")

        downloaded_paths.append({
            "path": file_path,
            "title": paper.title,
            "arxiv_id": paper.get_short_id(),
            "published": str(paper.published.date())
        })

    return downloaded_paths
    
fetch_arxiv_pdfs(ARXIV_IDS)