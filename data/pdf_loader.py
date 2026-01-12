import pymupdf as p
import os
from pathlib import Path
import json 

def load_pdf(folder_path):
    all_papers_data=[]
    papers_dir= Path(folder_path)

    for pdf_file in papers_dir.glob("*.pdf"):
        print(f"processing {pdf_file.name}")
        try:
            doc= p.open(pdf_file)
            paper_content={
                "filename": pdf_file.name,
                "pages": []
            }
            for page_num in range(len(doc)):
                page= doc[page_num]
                text= page.get_text()
                paper_content["pages"].append({
                    "page_number": page_num+1,
                    "text": text
                })
            
            all_papers_data.append(paper_content)
            doc.close()

        except Exception as e:
            print("Could not read pdfs.")

    return all_papers_data

papers_folder= "papers"
if __name__ == "__main__":
    all_data= load_pdf(papers_folder)

    # if all_data:
    #     first_paper_pages = all_data[0]["pages"]
    #     print(json.dumps(first_paper_pages, indent=4))

    # for paper in all_data:
    #     full_text = " ".join([page["text"] for page in paper["pages"]])
    #     words = full_text.split()
    #     preview = " ".join(words[:20])
    #     print(f"\nFile: {paper['filename']}")
    #     print(f"Preview: {preview}...")


    