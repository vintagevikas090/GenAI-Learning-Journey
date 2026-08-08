'''
    This File Includes the implementation for 
        1. Hybrid Search RAG (using PineCone)
        2. Ensemble RAG (How to use multiple retrivers)
        3. MultiQuery RAG (for better retrieval)
        4. Contextual Compression RAG (for handling large docs)
'''


###############################################################################
# Hybrid Search RAG
#
# What it does:
# Performs a combined query against a vector database (like Pinecone) that indexes both
# dense semantic vectors (capturing conceptual meaning) and sparse keyword vectors (like SPLADE/BM25,
# capturing exact matches), merging their relevance scores.
#
# Purpose:
# Uses Pinecone to perform hybrid search combining dense semantic embeddings and
# sparse lexical vectors.
#
# Use when:
# Applications require exact keyword matching (e.g. part numbers, codes) alongside
# conceptual similarity matching.
#
###############################################################################

# 1. Imports
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document

# 2. Configuration
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "your-groq-api-key")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY", "your-pinecone-api-key")

# 3. LLM
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

# 4. Embedding Model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 5. Load Documents
documents = [
    Document(page_content="Pinecone hybrid search merges dense vector index representations with sparse keyword index scores."),
]

# 6. Split Documents
chunks = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20).split_documents(documents)

# 7. Vector Store
# Initialize Pinecone hybrid vector index
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "hybrid-search-index"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

vectorstore = PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name)

# 8. Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 9. Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using context retrieved via Hybrid Search:\n\n{context}"),
    ("human", "{input}"),
])

# 10. Chain
document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

# 11. Invoke
response = rag_chain.invoke({"input": "What does Pinecone hybrid search merge?"})

# 12. Print Output
print(response["answer"])




###############################################################################
# Ensemble RAG
#
# What it does:
# Runs multiple distinct retrievers (in this case, BM25 keyword matching and FAISS semantic vector search)
# in parallel, and merges their ranked results using Reciprocal Rank Fusion (RRF) algorithm to generate a single context candidate list.
#
# Purpose:
# Combines search results from dense vector retrieval (FAISS) and sparse keyword
# retrieval (BM25) using Reciprocal Rank Fusion (RRF).
#
# Use when:
# Out-of-domain terms, technical jargon, or exact match queries need strong keyword
# search combined with vector semantic retrieval.
#
###############################################################################

# 1. Imports

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

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
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 2

faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Combine dense and sparse retrievers with custom weights
# weights defines the mathematical balance used in Reciprocal Rank Fusion (RRF) scoring.
# E.g. [0.5, 0.5] weights dense semantic match scores and sparse keyword match scores equally.
retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.5, 0.5]
)

# 9. Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using the context below:\n\n{context}"),
    ("human", "{input}"),
])

# 10. Chain
document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

# 11. Invoke
response = rag_chain.invoke({"input": "How does Ensemble RAG aggregate results?"})

# 12. Print Output
print(response["answer"])





###############################################################################
# MultiQuery RAG
#
# What it does:
# Uses an LLM to generate multiple versions of the user query, searches the vector DB
# with all generated variations, and combines the unique retrieved passages.
#
# Purpose:
# Automates prompt tuning by generating multiple perspectives of a single user input
# to retrieve a diverse set of relevant documents from vector storage.
#
# Use when:
# User queries are ambiguous, poorly framed, or missing specific keywords.
#
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

from langchain.retrievers import MultiQueryRetriever ####################

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
# Generate multiple query variations to improve retrieval recall
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)

# 9. Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using the retrieved context below:\n\n{context}"),
    ("human", "{input}"),
])

# 10. Chain
document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

# 11. Invoke
response = rag_chain.invoke({"input": "How do RAG systems improve LLM responses?"})

# 12. Print Output
print(response["answer"])



###############################################################################
# Contextual Compression RAG
#
# What it does:
# Passes the initially retrieved documents through a compressor (like an LLM chain or filter)
# that extracts only the exact relevant sentences or sub-passages matching the query,
# discarding irrelevant surrounding text before passing it to the final LLM prompt.
#
# Purpose:
# Extracts and compresses retrieved document passages using an LLM sequence filter
# before passing context to the final generation prompt.
#
# Use when:
# Documents are long, noisy, or contain irrelevant background text surrounding key facts.

###############################################################################

# 1. Imports

from langchain.retrievers import ContextualCompressionRetriever ##############
from langchain.retrievers.document_compressors import LLMChainExtractor  ##############

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
# Compress retrieved chunks using LLM extraction
compressor = LLMChainExtractor.from_llm(llm)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever()
)

# 9. Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using compressed context below:\n\n{context}"),
    ("human", "{input}"),
])

# 10. Chain
document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

# 11. Invoke
response = rag_chain.invoke({"input": "What does contextual compression do?"})

# 12. Print Output
print(response["answer"])





######### ADDITIONAL CONCEPTS ABOUT RETRIEVER #################


###############################################################################
# MMR
#
# Purpose:
# Maximal Marginal Relevance balances relevance to the query with diversity among
# selected retrieved chunks to eliminate duplicate information.
#
# Use when:
# Corpus contains redundant or highly repetitive text chunks.
###############################################################################

# Configure retriever with MMR search mode and lambda balance parameter
# fetch_k: the initial number of candidate documents to retrieve using semantic similarity
# lambda_mult: diversity weight (0.0 = maximum diversity/minimal similarity, 1.0 = maximum similarity/no diversity filter)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2, "fetch_k": 5, "lambda_mult": 0.7}
)


###############################################################################
# Similarity Score Threshold
#
# Purpose:
# Filters out retrieved chunks that do not meet a minimum cosine similarity confidence score.
#
# Use when:
# Preventing out-of-domain or low-relevance documents from polluting prompt context.
###############################################################################

# Filter retrieved results strictly based on minimum similarity score
# score_threshold: minimum similarity score cutoff (drops chunks below this score)
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.3}
)
