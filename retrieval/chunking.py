import json
import re
import nltk
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
import multiprocessing

def get_optimized_converter():
    cpu_count = multiprocessing.cpu_count()
    accel = AcceleratorOptions(num_threads=cpu_count, device=AcceleratorDevice.AUTO)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = accel
    
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = False
    
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading missing NLTK resources...")
    nltk.download('punkt')
    nltk.download('punkt_tab') 

INPUT_DIR = Path("data/raw_pdfs")
OUTPUT_FILE = Path("data/processed/knowledge_base_refined.jsonl")

TARGET_SIZE = 400  
MIN_SIZE = 150     
MAX_SIZE = 800     
OVERLAP_SENTENCES = 2 

GARBAGE_HEADERS = [
    r"^\d+$", r"^(19|20)\d{2}$", r"arXiv:\d+\.\d+", 
    r"(ICML|NeurIPS|ICLR|ACL|IEEE)\s?\d*", r"All rights reserved"
]



class SemanticChunker:
    def __init__(self):
        self.buffer_sentences = []
        self.current_word_count = 0
        self.chunks = []
        
    def add_sentence(self, sentence, metadata):
        word_count = len(sentence.split())
        
        self.buffer_sentences.append({
            "text": sentence,
            "metadata": metadata 
        })
        self.current_word_count += word_count
        
        if self.current_word_count >= TARGET_SIZE:
            self._flush_chunk(force=False)
            
    def _flush_chunk(self, force=False):
        if not self.buffer_sentences:
            return
        if not force and self.current_word_count < MIN_SIZE:
            return 

        full_text = " ".join([s["text"] for s in self.buffer_sentences])
        
        primary_meta = self.buffer_sentences[0]["metadata"]
        
        chunk_obj = {
            "text": full_text,
            "metadata": primary_meta,
            "stats": {
                "word_count": self.current_word_count,
                "sentence_count": len(self.buffer_sentences)
            }
        }
        self.chunks.append(chunk_obj)
        
        if not force:
            overlap = self.buffer_sentences[-OVERLAP_SENTENCES:]
            self.buffer_sentences = overlap
            self.current_word_count = sum(len(s["text"].split()) for s in overlap)
        else:
            self.buffer_sentences = []
            self.current_word_count = 0

    def finalize(self):
        self._flush_chunk(force=True)
        return self.chunks

def clean_text_noise(text):
    """Deep cleaning for text."""
    # Remove bracketed citations [1], [12]
    text = re.sub(r'\[\d+(?:,\s?\d+)*\]', '', text) 
    # Fix hyphenated line breaks (trans-\nformer -> transformer)
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    # Remove extra spaces
    return re.sub(r'\s+', ' ', text).strip()

def is_valid_header(text):
    if len(text) < 3: return False
    for pattern in GARBAGE_HEADERS:
        if re.search(pattern, text, re.IGNORECASE): return False
    return True

def enrich_table_markdown(table_md, metadata, caption=""):
    lines = table_md.strip().split('\n')
    columns = "Unknown columns"
    
    # Heuristic: Find the first line starting with | that isn't a separator |---|
    for line in lines:
        if line.strip().startswith('|') and '---' not in line:
            # Extract words between pipes
            cols = [c.strip() for c in line.split('|') if c.strip()]
            if cols:
                columns = ", ".join(cols)
            break
            
    # construct Natural Language Context
    header = (
        f"### Table Context\n"
        f"**Source Paper:** {metadata['source_title']}\n"
        f"**Section:** {metadata['section']}\n"
        f"**Table Caption:** {caption if caption else 'No caption detected'}\n"
        f"**Columns:** {columns}\n"
        f"**Content Summary:** This table presents data regarding {columns} in the context of {metadata['section']}.\n"
        f"--------------------------------------------------\n"
    )
    
    return header + table_md

# --- CHANGE: Replace the entire run_chunking() function with this ---

def process_single_pdf(pdf_path: Path):
    """
    Processes a single PDF and returns a list of dictionaries (chunks).
    This is called by your Streamlit app.
    """
    # 1. Setup Docling (Moved inside to ensure fresh state for each dynamic upload)
    pipeline_options = PdfPipelineOptions(do_table_structure=True)
    pipeline_options.table_structure_options.do_cell_matching = True
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    print(f"Processing: {pdf_path.name}")
    try:
        result = converter.convert(pdf_path)
        doc = result.document
    except Exception as e:
        print(f"!! Error parsing: {e}")
        return []

    # 2. Extract Metadata (Handling dynamic filenames from Streamlit)
    # If the filename is "2401.00396_RAGTruth.pdf", split it. 
    # Otherwise, use the whole filename as the title.
    file_parts = pdf_path.stem.split("_", 1)
    arxiv_id = file_parts[0] if len(file_parts) > 1 else "upload"
    doc_title = file_parts[1] if len(file_parts) > 1 else pdf_path.stem

    chunker = SemanticChunker()
    current_section = "Abstract"
    previous_text_element = "" 
    paper_knowledge_base = [] # Local list for this specific paper

    # 3. The Core Loop (Your existing logic, slightly adjusted)
    for item_ref in doc.body.children:
        try:
            element = item_ref.resolve(doc)
        except AttributeError: continue
        if not hasattr(element, "label"): continue

        # HEADERS logic
        if element.label == "section_header":
            text = element.text.strip()
            if any(x in text.lower() for x in ["references", "bibliography"]):
                break 
            if is_valid_header(text):
                chunker._flush_chunk(force=True) 
                current_section = text
            continue

        # TABLES logic
        if element.label == "table":
            chunker._flush_chunk(force=True)
            table_md = element.export_to_markdown(doc=doc)
            caption = previous_text_element if previous_text_element.strip().lower().startswith("table") else ""
            
            meta = {
                "source_id": arxiv_id,
                "source_title": doc_title,
                "section": current_section,
                "type": "table",
                "page": element.prov[0].page_no if element.prov else 0
            }

            paper_knowledge_base.append({
                "chunk_id": f"{arxiv_id}_tb_{len(paper_knowledge_base)}",
                "text": enrich_table_markdown(table_md, meta, caption),
                "metadata": meta
            })
            previous_text_element = ""
            continue

        # TEXT logic
        if element.label == "text":
            raw_text = clean_text_noise(element.text)
            if not raw_text or len(raw_text) < 10: continue
            previous_text_element = raw_text 

            sentences = nltk.sent_tokenize(raw_text)
            for sent in sentences:
                meta = {
                    "source_id": arxiv_id,
                    "source_title": doc_title,
                    "section": current_section,
                    "type": "text",
                    "page": element.prov[0].page_no if element.prov else 0
                }
                chunker.add_sentence(sent, meta)

    # 4. Finalize and return
    paper_text_chunks = chunker.finalize()
    for i, chunk in enumerate(paper_text_chunks):
        chunk["chunk_id"] = f"{arxiv_id}_tx_{len(paper_knowledge_base) + i}"
    
    paper_knowledge_base.extend(paper_text_chunks)
    return paper_knowledge_base