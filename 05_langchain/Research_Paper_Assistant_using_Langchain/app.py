import streamlit as st
import os
from src.vector_store import create_vector_store
from src.rag_chain import create_rag_chain

folder_path = "papers"
os.makedirs(folder_path, exist_ok=True)

st.set_page_config(
    page_title="Research Paper Assistant",
    layout="centered"
)

# session chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

@st.cache_resource
def build_rag_system(folder_path):
    vector_store = create_vector_store(folder_path)
    retrieval_chain = create_rag_chain(vector_store)
    return retrieval_chain

st.title("📚 Research Paper Assistant")

################### SIDEBAR #####################################

with st.sidebar:
    st.header("💬 Conversation History")
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**🧑 You:** {message['content']}")
            else:
                st.markdown(f"**🤖 Assistant:** {message['content']}")
                st.divider()



################## File Upload Section ###########################
uploaded_files = st.file_uploader("📤 Upload the Research Papers",type="pdf",accept_multiple_files = True)

if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(folder_path, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

    build_rag_system.clear()
    st.success(f"✅ {len(uploaded_files)} Paper(s) Uploaded Successfully")



##################### Show Available PDFs ########################

pdf_files = [ file for file in os.listdir(folder_path) if file.endswith(".pdf")]
if pdf_files:
    st.subheader("📑 Available Research Papers")
    st.dataframe(pdf_files,width="stretch")
    

################### Process the Papers ###########################

col1, col2 = st.columns(2)

with col1:
    process_button = st.button( "⚙️ Process Papers", use_container_width=True)

if process_button:
    if not pdf_files:
        st.warning("⚠️ Please Upload at least one Research Paper")
    else:
        with st.spinner("⚙️ Processing Papers..."):
            retrieval_chain = build_rag_system(folder_path)
            st.session_state["retrieval_chain"] = retrieval_chain

        st.success("✅ Research Papers Successfully Processed")




##################### Summarize Papers ###########################
with col2:
    summarize_button = st.button("📝 Summarize Papers",use_container_width=True)

summary_query = """
    Create a detailed summary of the research papers.
        Use the following structure:
            - Objective
            - Background
            - Methodology
            - Key Findings
            - Conclusion
            - Future Scope
    Only use information available in the uploaded papers.
    """

if summarize_button:
    if 'retrieval_chain' not in st.session_state:
        st.warning('⚠️ Please Process the Papers First')
    else:
        with st.spinner("📝 Generating Summary..."):
            ret_chain = st.session_state["retrieval_chain"]
            res = ret_chain.invoke({ "input": summary_query })

        st.success('✅ Summery Generated Successfully')
        st.subheader('Research Paper Summery')
        st.write(res["answer"])



################### Query Section ################################
if "retrieval_chain" in st.session_state:
    query = st.chat_input("💡 Ask a question about the papers")

    if query:
        with st.spinner("🤔 Thinking..."):
            ret_chain = st.session_state["retrieval_chain"]
            res = ret_chain.invoke({"input": query})

        answer = res["answer"]

        # Store user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": query
        })

        # Store assistant message
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })

        st.rerun()




################### Current Answer ###############################

if st.session_state.chat_history:
    last_assistant_message = ''
    for message in reversed(st.session_state.chat_history):
        if message["role"] == "assistant":
            last_assistant_message = message["content"]
            break

    if last_assistant_message:
        st.subheader("🤖 Answer")
        st.write(last_assistant_message)

