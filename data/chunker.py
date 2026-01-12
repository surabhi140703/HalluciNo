from pdf_loader import load_pdf
import json 
import re 

def get_sections_and_chunks(papers_data, chunk_size=300, overlap=100):
    all_chunks=[]
    global_chunk_count=1
    header_pattern= re.compile(r'^(\d+\.?\s+[A-Z][a-z]+|[A-Z]{2,}|Abstract|References|Introduction)', re.MULTILINE)

    for paper in papers_data:
        print(f"--- Processing paper: {paper['filename']} ---")
        current_section= "start matter"

        for page in paper["pages"]:
            text= page["text"]
            page_num= page["page_number"]

            parts= header_pattern.split(text)

            for part in parts:
                part = part.strip()
                if not part: 
                    continue

                if header_pattern.match(part):
                    current_section= part
                    continue

                words= part.split()
                for j in range(0, len(words), chunk_size- overlap):
                    chunk_words= words[j: j+chunk_size]
                    if len(chunk_words)<10: 
                        continue

                    all_chunks.append({
                        "chunk_no": global_chunk_count,
                        "source": paper["filename"],
                        "page": page_num,
                        "section": current_section,
                        "text": " ".join(chunk_words)
                    })

                    global_chunk_count+=1
    return all_chunks

def save_to_jsonl(chunks,output_path):
    with open(output_path,'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk)+ '\n')

if __name__ == "__main__":
    raw_data= load_pdf("papers")
    processed_chunks= get_sections_and_chunks(raw_data)
    save_to_jsonl(processed_chunks, "processed_data.jsonl")
