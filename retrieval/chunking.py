import json
import re
import nltk
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

# Download NLTK tokenizer (run once)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading missing NLTK resources...")
    nltk.download('punkt')
    nltk.download('punkt_tab')  # <--- THIS WAS MISSING

# --- CONFIGURATION ---
INPUT_DIR = Path("data/raw_pdfs")
OUTPUT_FILE = Path("data/processed/knowledge_base_refined.jsonl")

# Semantic Constraints
TARGET_SIZE = 400  # distinct words
MIN_SIZE = 150     # If smaller, merge with next
MAX_SIZE = 800     # Hard limit
OVERLAP_SENTENCES = 2 # Number of sentences to carry over

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
        """Accumulates sentences until TARGET_SIZE is reached."""
        word_count = len(sentence.split())
        
        self.buffer_sentences.append({
            "text": sentence,
            "metadata": metadata # Store metadata per sentence to handle section transitions
        })
        self.current_word_count += word_count
        
        # Trigger chunk creation if target reached
        if self.current_word_count >= TARGET_SIZE:
            self._flush_chunk(force=False)
            
    def _flush_chunk(self, force=False):
        """Finalizes the buffer into a chunk."""
        if not self.buffer_sentences:
            return

        # 1. Check constraints
        if not force and self.current_word_count < MIN_SIZE:
            return # Keep building
            
        # 2. Consolidate Text
        full_text = " ".join([s["text"] for s in self.buffer_sentences])
        
        # 3. Consolidate Metadata (Use the metadata of the majority of sentences)
        # (Simplification: Use the metadata of the first sentence for the 'Source')
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
        
        # 4. Handle Overlap (Keep last N sentences)
        if not force:
            overlap = self.buffer_sentences[-OVERLAP_SENTENCES:]
            self.buffer_sentences = overlap
            self.current_word_count = sum(len(s["text"].split()) for s in overlap)
        else:
            self.buffer_sentences = []
            self.current_word_count = 0

    def finalize(self):
        """Force flush any remaining text."""
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
    """
    Creates a 'Searchable Header' for the table.
    It extracts column names from the markdown to help the vector store.
    """
    # 1. Parse Markdown to find Headers
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
            
    # 2. Construct Natural Language Context
    # This string is what the Embedding Model will "Read"
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

def run_chunking():
    print("Initializing Docling Layout Engine...")
    pipeline_options = PdfPipelineOptions(do_table_structure=True)
    pipeline_options.table_structure_options.do_cell_matching = True
    
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    all_pdfs = list(INPUT_DIR.glob("*.pdf"))
    final_knowledge_base = []

    for pdf_path in all_pdfs:
        print(f"\nProcessing: {pdf_path.name}")
        try:
            result = converter.convert(pdf_path)
            doc = result.document
        except Exception as e:
            print(f"!! Error parsing: {e}")
            continue

        # Meta extraction
        file_parts = pdf_path.stem.split("_", 1)
        arxiv_id = file_parts[0]
        doc_title = file_parts[1] if len(file_parts) > 1 else pdf_path.stem

        # State Tracking
        chunker = SemanticChunker() # [Ensure your SemanticChunker class is defined above]
        current_section = "Abstract"
        previous_text_element = "" 
        
        # Iterate over Structure
        for item_ref in doc.body.children:
            try:
                element = item_ref.resolve(doc)
            except AttributeError: continue
            if not hasattr(element, "label"): continue

            # --- 1. HEADERS ---
            if element.label == "section_header":
                text = element.text.strip()
                if any(x in text.lower() for x in ["references", "bibliography"]):
                    print("  -> Bibliography reached. Finalizing paper.")
                    break 
                
                if is_valid_header(text):
                    chunker._flush_chunk(force=True) 
                    current_section = text
                continue

            # --- 2. TABLES (OPTIMIZED FOR RETRIEVAL) ---
            if element.label == "table":
                # Flush text buffer so table stands alone
                chunker._flush_chunk(force=True)
                
                # A. Get Raw Markdown (Pass doc=doc to fix warning)
                table_md = element.export_to_markdown(doc=doc)
                
                # B. Find Caption (Look back at previous text)
                caption = ""
                if previous_text_element.strip().lower().startswith("table"):
                    caption = previous_text_element.strip()

                # C. Build Metadata
                meta = {
                    "source_id": arxiv_id,
                    "source_title": doc_title,
                    "section": current_section,
                    "type": "table",
                    "page": element.prov[0].page_no if element.prov else 0
                }

                # D. ENRICH CONTENT (The Fix)
                # We inject the "Searchable Header" directly into the text field
                searchable_text = enrich_table_markdown(table_md, meta, caption)

                final_knowledge_base.append({
                    "chunk_id": f"{arxiv_id}_tb_{len(final_knowledge_base)}",
                    "text": searchable_text, # <--- Now contains Context + Columns + Table
                    "metadata": meta
                })
                
                print(f"  -> Captured Table with columns from section: {current_section}")
                previous_text_element = ""
                continue

            # --- 3. TEXT ---
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

        # Final flush
        paper_chunks = chunker.finalize()
        final_knowledge_base.extend(paper_chunks)

    # Save
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for chunk in final_knowledge_base:
            f.write(json.dumps(chunk) + '\n')
            
    print(f"\nSaved {len(final_knowledge_base)} chunks to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_chunking()