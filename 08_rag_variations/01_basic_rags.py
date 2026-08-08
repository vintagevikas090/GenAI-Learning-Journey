'''
    This File Includes the implementation for 
        1. Basic RAG (without Memory)
        2. Conversational RAG (With Memory)
        3. History Aware RAG
'''


###############################################################################
# Basic RAG
#
# Purpose:
# Standard Retrieval-Augmented Generation pipeline using LCEL to answer queries
# grounded in retrieved vector database context.
#
# Use when:
# Single-turn Q&A over custom domain documents without conversational memory.
###############################################################################

# 1. Imports
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

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
    ("system", "Answer the user question strictly using the provided context:\n\n{context}"),
    ("human", "{input}"),
])

# 10. Chain
document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

# 11. Invoke
response = rag_chain.invoke({"input": "What is LangChain?"})

# 12. Print Output
print(response["answer"])


###############################################################################
# Conversational RAG
#
# Purpose:
# Multi-turn RAG pipeline that maintains interaction context across turns alongside
# document retrieval.
#
# Use when:
# Building interactive chatbots or assistants requiring persistent chat history.
###############################################################################

# 1. Imports
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

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
    ("system", "Answer the user question using the context below:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# 10. Chain
document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

# 11. Invoke
chat_history = [
    HumanMessage(content="Hi, I am researching AI coding tools."),
    AIMessage(content="Hello! I can help answer questions regarding AI coding tools."),
]
response = rag_chain.invoke({
    "chat_history": chat_history,
    "input": "What is Antigravity?",
})

# 12. Print Output
print(response["answer"])


###############################################################################
# History-Aware RAG
#
# Purpose:
# Re-formulates follow-up queries into standalone questions before retrieval to ensure
# accurate vector similarity search across conversational turns.
#
# Use when:
# Multi-turn conversations where user follow-up questions reference previous turns
# using pronouns or ambiguous context (e.g. "What about its key features?").
###############################################################################

# 1. Imports

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage

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
# Formulate standalone query from conversation history before vector search
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
retriever = create_history_aware_retriever(llm, vectorstore.as_retriever(search_kwargs={"k": 2}), contextualize_q_prompt)

# 9. Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using the context below:\n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# 10. Chain
document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

# 11. Invoke
chat_history = [
    HumanMessage(content="What programming language released version 3.12?"),
    AIMessage(content="Python released version 3.12."),
]
response = rag_chain.invoke({
    "chat_history": chat_history,
    "input": "What are its key features?",
})

# 12. Print Output
print(response["answer"])
