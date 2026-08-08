'''
    This File Includes the implementation for 
        1. Corrective RAG (using Document Grader)
        2. Self RAG (using Answer Grader)
'''


###############################################################################
# CRAG
#
# Purpose:
# Corrective RAG evaluates document relevance after retrieval and uses web search or query
# re-writing as a fallback mechanism when retrieved chunks are inadequate.
#
# Use when:
# Unreliable vector store search results require validation before answer generation.
###############################################################################

# 1. Imports
import os
from typing import TypedDict, List
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langgraph.graph import END, StateGraph, START

# 2. Configuration
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "your-groq-api-key")

# 3. LLM
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

# 4. Embedding Model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 5. Load Documents
documents = PyPDFLoader("sample.pdf").load()

# 6. Split Documents
chunks = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20).split_documents(documents)

# 7. Vector Store
vectorstore = FAISS.from_documents(chunks, embeddings)

# 8. Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 9. Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using the context below:\n\n{context}"),
    ("human", "{input}"),
])

# 10. Chain
document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

# Grader schema used to evaluate document relevance binary score.
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")

# Grader prompt instructing LLM to assess document relevance to the query.
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a grader assessing relevance of a document to a user question. Grade 'yes' if relevant, else 'no'."),
    ("human", "Retrieved Document:\n{context}\n\nUser Question: {question}"),
])

####### Create structured grader chain -> for GRADING THE DOCS
grader_chain = grade_prompt | llm.with_structured_output(GradeDocuments) 


# Define LangGraph State dictionary that will flow through graph nodes.
class GraphState(TypedDict):
    question: str
    documents: List[str]
    generation: str

# Node 1: Retrieval step to fetch documents from vector DB.
def retrieve_node(state: GraphState):
    docs = retriever.invoke(state["question"])
    return {"documents": [d.page_content for d in docs]}

# Node 2: Grading step. Loops over documents, scoring each document. Filters out irrelevant ones.
def grade_node(state: GraphState):
    relevant_docs = []
    for doc in state["documents"]:
        res = grader_chain.invoke({"question": state["question"], "context": doc})
        if res.binary_score.lower() == "yes":
            relevant_docs.append(doc)
    return {"documents": relevant_docs} ##### REPLACING current docs with the RELEVANT DOCS

# Node 3: Generation step. Constructs prompt using GRADED RELEVANT documents.
def generate_node(state: GraphState):
    context = "\n".join(state["documents"]) if state["documents"] else "No relevant context found."
    out = llm.invoke(f"Question: {state['question']}\nContext: {context}")
    return {"generation": out.content}

# Compile the LangGraph state machine.
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_node)
workflow.add_node("generate", generate_node)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_edge("grade", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# 11. Invoke
response = app.invoke({"question": "What is Corrective RAG?"})

# 12. Print Output
print(response["generation"])


###############################################################################
# Self-RAG
#
# Purpose:
# Self-Reflective RAG iteratively evaluates generation quality, checking for hallucinations
# and question coverage to regenerate or re-retrieve when necessary.
#
# Use when:
# High precision applications requiring zero hallucination tolerance and verified output.
###############################################################################

# 1. Imports
import os
from typing import TypedDict, List
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langgraph.graph import END, StateGraph, START

# 2. Configuration
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "your-groq-api-key")

# 3. LLM
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

# 4. Embedding Model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 5. Load Documents
documents = PyPDFLoader("sample.pdf").load()

# 6. Split Documents
chunks = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20).split_documents(documents)

# 7. Vector Store
vectorstore = FAISS.from_documents(chunks, embeddings)

# 8. Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 9. Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question strictly using the provided context:\n\n{context}"),
    ("human", "{input}"),
])

# 10. Chain
document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

# Schema for grading hallucination. Verifies if LLM generation matches retrieved facts.
class GradeHallucination(BaseModel):
    binary_score: str = Field(description="Grade 'yes' if generation is grounded in context, else 'no'")

# Hallucination prompt instructing the LLM to verify factual grounding of generation to context.
hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", "Grade whether the generation is grounded in facts presented in context. Respond 'yes' or 'no'."),
    ("human", "Context: {context}\n\nGeneration: {generation}"),
])
hallucination_grader = hallucination_prompt | llm.with_structured_output(GradeHallucination)


# Self-RAG graph state definition.
class SelfRAGState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    is_grounded: str

# Node 1: Fetch candidate chunks.
def self_retrieve(state: SelfRAGState):
    docs = retriever.invoke(state["question"])
    return {"documents": [d.page_content for d in docs]}

# Node 2: Generate response using documents.
def self_generate(state: SelfRAGState):
    context = "\n".join(state["documents"])
    res = rag_chain.invoke({"input": state["question"], "context": context})
    return {"generation": res["answer"]}

# Node 3: Fact-check step. Checks for hallucination by grading generated answer vs context.
def self_check_hallucination(state: SelfRAGState):
    context = "\n".join(state["documents"])
    check = hallucination_grader.invoke({"context": context, "generation": state["generation"]})
    return {"is_grounded": check.binary_score}

# Build workflow state machine.
self_workflow = StateGraph(SelfRAGState)
self_workflow.add_node("retrieve", self_retrieve)
self_workflow.add_node("generate", self_generate)
self_workflow.add_node("check_hallucination", self_check_hallucination)

self_workflow.add_edge(START, "retrieve")
self_workflow.add_edge("retrieve", "generate")
self_workflow.add_edge("generate", "check_hallucination")
self_workflow.add_edge("check_hallucination", END)

app = self_workflow.compile()

# 11. Invoke
response = app.invoke({"question": "What does Self-RAG introduce?"})

# 12. Print Output
print(response["generation"])
