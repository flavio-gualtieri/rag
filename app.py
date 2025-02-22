import streamlit as st
import os
from tools import Tools
from rag import RAG
import tempfile

# Initialize tools and RAG
st.title("Chat with Your PDF")

tools = Tools(chunk_size=800, overlap=100)
rag = RAG(tools)

col1, col2 = st.columns([3,1])

with col1:
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_path = tmp_file.name
        
        st.write("Processing PDF...")
        document_name = os.path.basename(file_path)
        chunk_df = rag.process_pdf(file_path, document_name)

        if chunk_df is not None:
            st.success("PDF processed successfully! You can now start chatting.")
            st.session_state["document_name"] = document_name  # Store for later queries
        else:
            st.error("Failed to process the PDF.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat history
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User Input Section
    if uploaded_file:  # Only allow chat after file is uploaded
        user_input = st.chat_input("Ask a question about the document...")

        if user_input:
            # Display user message
            st.session_state["messages"].append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            # Generate response
            with st.spinner("Retrieving relevant information..."):
                document_name = st.session_state.get("document_name", "")
                answer = rag.generate_answer(user_input, document_name)

            # Display assistant response
            with st.chat_message("assistant"):
                st.write(answer)

            # Store assistant message
            st.session_state["messages"].append({"role": "assistant", "content": answer})