# 📄 PaperLens

An AI-powered Research Paper Assistant built using LangChain, FAISS, Ollama, and Streamlit. PaperLens enables users to upload research papers, generate concise summaries, and ask questions about document content through a Retrieval-Augmented Generation (RAG) pipeline running entirely on local models.

---

## 🚀 Overview

PaperLens is designed to simplify research paper exploration by combining semantic search with large language models. Instead of manually reading lengthy documents, users can interact with their papers conversationally and receive context-aware answers grounded in the uploaded content.

The application uses a local-first architecture powered by Ollama, ensuring privacy and eliminating the need for paid API services.

---

## ✨ Features

* 📄 Upload one or more research papers in PDF format
* ✂️ Automatic document chunking for efficient retrieval
* 🧠 Local embedding generation using Ollama
* 🔍 Semantic search with FAISS vector database
* 📚 Retrieval-Augmented Generation (RAG)
* ❓ Ask questions about uploaded papers
* 📝 Generate structured document summaries
* 💬 Interactive chat interface
* 🔄 Session-based conversation history
* 💻 Fully local execution
* 🔒 No external API keys required

---

## 🏗️ System Architecture

```text
Research Papers (PDFs)
          │
          ▼
      PyPDFLoader
          │
          ▼
      Documents
          │
          ▼
RecursiveCharacterTextSplitter
          │
          ▼
      Text Chunks
          │
          ▼
 Ollama Embeddings
(nomic-embed-text)
          │
          ▼
        FAISS
          │
          ▼
      Retriever
          │
          ▼
 Retrieved Chunks
          │
          ▼
       Gemma 2B
        (LLM)
          │
          ▼
 Answer / Summary
```

---

## 🛠️ Tech Stack

| Category            | Technology                     |
| ------------------- | ------------------------------ |
| **Frontend**        | Streamlit                      |
| **Framework**       | LangChain                      |
| **LLM**             | Gemma 2B (Ollama)              |
| **Embeddings**      | nomic-embed-text               |
| **Vector Database** | FAISS                          |
| **Document Loader** | PyPDFLoader                    |
| **Text Splitting**  | RecursiveCharacterTextSplitter |

---

## 📚 Key Concepts Demonstrated

This project showcases several fundamental Generative AI and RAG concepts:

* Retrieval-Augmented Generation (RAG)
* Semantic Search
* Vector Databases
* Embedding Models
* Document Processing Pipelines
* Prompt Engineering
* Retrieval Chains
* PDF Knowledge Extraction
* Local LLM Inference with Ollama
* Streamlit Application Development

---

## 📁 Project Structure

```text
PaperLens/
│
├── app.py
│
├── preview/
│
├── papers/
│
├── functions/
│   ├── rag_chain.py
│   └── vector_store.py
│
├── requirements.txt
├── README.md
```

---

## ⚙️ Prerequisites

Install Ollama from:

https://ollama.com

Pull the required models:

```bash
ollama pull gemma:2b
```

```bash
ollama pull nomic-embed-text
```

Start the Ollama server:

```bash
ollama serve
```

---

## 🔧 Installation

Clone the repository:

```bash
git clone https://github.com/vintagevikas090/PaperLens.git
```

Move into the project directory:

```bash
cd PaperLens
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

The application will launch in your browser.

---

## 💡 Example Questions

* Summarize this research paper.
* What is the main objective of the study?
* What methodology was used?
* What are the key findings?
* What limitations are discussed?
* Explain the results in simple terms.
* What future work is suggested by the authors?

---

## 🔮 Future Improvements

* Conversational RAG with memory
* Multi-document knowledge base
* Hybrid Search (Dense + Sparse Retrieval)
* Source citations and page references
* Research paper comparison
* LangGraph-based workflows
* Agentic document analysis
* Export summaries and notes

---

## 👨‍💻 Author

**Vikas Prajapat**

Applied AI / Generative AI Engineer

Currently exploring LangChain, RAG Systems, LangGraph, MCP, Agentic AI, and Production-Ready GenAI Applications.
