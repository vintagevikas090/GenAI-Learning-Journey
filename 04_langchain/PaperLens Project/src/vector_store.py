from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

import os


def create_vector_store(folder_path):

    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)

            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

    chunks = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vector_store = FAISS.from_documents(documents=chunks,embedding=embeddings)

    return vector_store