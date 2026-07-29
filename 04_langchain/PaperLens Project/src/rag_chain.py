from langchain_ollama import OllamaLLM
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


def create_rag_chain(vector_store):

    llm = OllamaLLM(model="gemma:2b")

    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs ={"fetch_k": 10})

    prompt = ChatPromptTemplate.from_template(
                """
        You are a helpful AI research assistant.
        Answer the question only from the provided context.
        <context>
            {context}
        </context>

        Question: {input}
        """
            )

    document_chain = create_stuff_documents_chain(llm,prompt)

    retrieval_chain = create_retrieval_chain(retriever,document_chain)

    return retrieval_chain