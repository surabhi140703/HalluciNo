from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field 

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

#few shot prompting to ensure llm response complies with pydantic parser
grader_system_prompt = """
You are a grader assessing relevance of a retrieved document to a user question. \n 
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
Follow the pattern of the examples below. Output ONLY the JSON.

EXAMPLES:
Input Question: "How do Transformers work?"
Input Document: "The Transformer model uses self-attention mechanisms to process sequences."
Output: {{"binary_score": "yes"}}

Input Question: "What is the capital of France?"
Input Document: "The recipe for chocolate cake includes flour, sugar, and cocoa."
Output: {{"binary_score": "no"}}
"""

grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", grader_system_prompt),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)