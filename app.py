import streamlit as st
import tempfile
from pathlib import Path
from retrieval.agent_logic import app 
from retrieval.chunking import process_single_pdf
from retrieval.pipeline import RAGPipeline

st.set_page_config(page_title="HalluciNo Researcher", layout="wide")

if "rag" not in st.session_state:
    st.session_state.rag = RAGPipeline()

st.title("🤖 HalluciNo: Agentic Research Assistant")


with st.sidebar:
    st.header("Control Panel")
    uploaded_files = st.file_uploader(
        "Upload Research Papers", 
        type="pdf", 
        accept_multiple_files=True
    )

    if uploaded_files and st.button("🚀 Index All Papers"):
        total_chunks = 0
        progress_bar = st.progress(0)
        
        with st.status("Batch Processing Papers...", expanded=True) as status:
            for i, uploaded_file in enumerate(uploaded_files):
                status.write(f"📄 Processing: **{uploaded_file.name}**")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = Path(tmp.name)
                
                try:
                    new_chunks = process_single_pdf(tmp_path)
                    st.session_state.rag.add_new_paper(new_chunks)
                    
                    total_chunks += len(new_chunks)
                    status.write(f"✅ Added {len(new_chunks)} chunks.")
                except Exception as e:
                    st.error(f"Error indexing {uploaded_file.name}: {e}")
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink() 
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status.update(label=f"Successfully indexed {len(uploaded_files)} papers!", state="complete")
        st.toast(f"Knowledge base updated: {total_chunks} chunks.")

    st.divider()
    show_reasoning = st.checkbox("Show Agent Reasoning", value=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask a question about the papers...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        if show_reasoning:
            status_container = st.status("Agent is thinking...", expanded=True)
        
        try:
            inputs = {"question": user_input}
            
            for output in app.stream(inputs):
                for key, value in output.items():
                    if show_reasoning:
                        if key == "retrieve":
                            status_container.write(f"📚 Found {len(value['documents'])} potential chunks.")
                        elif key == "grade_documents":
                            status_container.write(f"✅ Grader verified {len(value['documents'])} chunks as relevant.")
                        elif key == "generate":
                            status_container.write("🧠 Synthesizing comparative answer...")
                    if "generation" in value:
                        full_response = value["generation"]

            if show_reasoning:
                status_container.update(label="Research Complete", state="complete", expanded=False)
            
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Agent Error: {e}")