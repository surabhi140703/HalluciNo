from pdf_loader import load_pdf
import json 
import re 

def get_sections_and_chunks(papers_data, chunk_size=150, overlap=30):
    all_chunks=[]
    header_pattern= re.compile(r'^(\d+\.? \s+ [A-Z][a-z]+|[A-Z]{2,}|Abstract|References|Introduction)', re.MULTILINE)

    for paper in papers_data:
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
                        "source": paper["filename"],
                        "page": page_num,
                        "section": current_section,
                        "text": " ".join(chunk_words)
                    })
    return all_chunks

if __name__ == "__main__":
    raw_data= load_pdf("papers")
    processed_chunks= get_sections_and_chunks(raw_data)
    print(json.dumps(processed_chunks[:7], indent=4))
