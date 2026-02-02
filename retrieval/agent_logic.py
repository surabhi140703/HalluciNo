from typing import List, Dict, Any
from typing_extensions import TypedDict
from langchain.schema import Document 
from langgraph.graph import StateGraph, END 

from retrieval.pipeline import RAGPipeline
from retrieval.prompts import grader_prompt, GradeDocuments

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
llm= ChatOllama(model="llama3.2", temperature=0, base_url="http://localhost:11434", format="json")
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser(pydantic_object=GradeDocuments)
grader_chain = grader_prompt | llm | parser
print("Initialising RAG Pipeline for agent:")
rag_tool= RAGPipeline()

class AgentState(TypedDict):
    question : str
    documents: List[Dict[str, Any]]
    generation: str

def retrieve_node(state):
    print("Retrieving: ")
    question= state["question"]
    documents= rag_tool.retrieve(question, top_k=25)
    return {"documents": documents, "question": question}



def grade_documents_node(state):
    print("Checking relevance:")
    question = state["question"]
    documents = state["documents"]
    
    filtered_docs = []
    
    for doc in documents:
        doc_text = f"Title: {doc['metadata']['source_title']}\nContent: {doc['text']}"
        
        try:
            score = grader_chain.invoke({"question": question, "document": doc_text})
            print(f"RAW LLM OUTPUT: {score}")
            grade = normalize_grade(score)
            
            if grade and grade.lower() == "yes":
                print(f"---GRADE: RELEVANT ({doc['metadata']['source_title']})---")
                filtered_docs.append(doc)
            elif grade and grade.lower() == "no":
                print(f"---GRADE: NOT RELEVANT ({doc['metadata']['source_title']})---")
            else:
                print(f"---GRADE WARNING: Could not decipher grade. Defaulting to RELEVANT.---")
                filtered_docs.append(doc)
                
        except Exception as e:
            print(f"---GRADE ERROR: {e}. Keeping doc.---")
            filtered_docs.append(doc)
            
    return {"documents": filtered_docs, "question": question}

def normalize_grade(score_dict):
    if 'yes' in score_dict:
        return 'yes'
    if 'no' in score_dict:
        return 'no'

    if 'binary_score' in score_dict:
        return score_dict['binary_score']
    if 'relevance' in score_dict:
        return score_dict['relevance']
    if 'score' in score_dict:
        return score_dict['score']
            
    return None

def generate_node(state):
    print("Generating:")
    question = state["question"]
    documents = state["documents"]
    
    context_str = ""
    for i, chunk in enumerate(documents):
        context_str += f"""
        [Document {i+1}]: {chunk['metadata']['source_title']} (Section: {chunk['metadata']['section']})
        Content: {chunk['text']}
        \n
        """

    system_prompt_content = """
    You are a senior academic researcher. Your goal is to synthesize answers from the provided research papers.

    INSTRUCTIONS:
    1. Answer the user's question using ONLY the provided documents.
    2. If the documents mention different approaches (e.g., Paper A says X, Paper B says Y), explicitly compare them.
    3. If the documents do not contain the answer, admit it. Do not hallucinate.

    CITATION RULES:
    - Start every claim with the bolded paper title. Example: **[Paper Title]** claims that...
    - Do NOT use JSON formatting.
    - Do NOT use code blocks.
    - Write in plain, professional English.

    FORMAT:
    - **Synthesis**: [Your direct answer comparing the papers]
    - **Key Evidence**: [Bulleted list of findings]
    """

    gen_llm = ChatOllama(
        model="llama3.2", 
        temperature=0.1, #
        base_url="http://localhost:11434"
    )
    
    messages = [
        SystemMessage(content=system_prompt_content),
        HumanMessage(content=f"Research Documents:\n{context_str}\n\nQuestion: {question}")
    ]
    
    response = gen_llm.invoke(messages)
    return {"generation": response.content}
def refusal_node(state):
    print("Refusal.")
    return {"generation": "Sorry, out of syllabus. (No relevant documents found in the knowledge base)."}

def decide_to_generate(state):
    filtered_documents= state["documents"]
    if not filtered_documents:
        return "refuse"
    else:
        return "generate"
    
#LANGGRAPH PART
workflow= StateGraph(AgentState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("generate", generate_node)
workflow.add_node("refusal", refusal_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve","grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate": "generate",
        "refuse": "refusal"
    }
)
workflow.add_edge("generate", END)
workflow.add_edge("refusal",END)

app= workflow.compile()

if __name__ == "__main__":
    test_question = "What is the relationship between 'RAGTruth's' fine-tuning results and the 'Semantic Illusion's' solution?"
    
    print(f"testing agent with: {test_question}\n")
    inputs = {"question": test_question}
    
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Finished Node: {key}")
            if "generation" in value:
                print("\nFINAL ANSWER:\n", value["generation"])