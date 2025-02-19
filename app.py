import streamlit as st
import os
from tools import Tools
from rag import RAG
import tempfile

# Initialize tools and RAG
st.title("Chat with Your PDF")

tools = Tools(chunk_size=800, overlap=100)
rag = RAG(tools)

# File Upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name
    
    st.write("Processing PDF...")
    chunk_df = rag.process_pdf(file_path)
    if chunk_df is not None:
        st.success("PDF processed successfully! You can now ask questions.")
    else:
        st.error("Failed to process the PDF.")

# User Question Input
question = st.text_input("Ask a question about the document:")
if question:
    with st.spinner("Retrieving relevant information..."):
        answer = rag.generate_answer(question)
    
    st.subheader("Answer:")
    st.write(answer)

    # Show relevant chunks for transparency
    relevant_chunks = rag.retrieve_relevant_chunks(question)
    if relevant_chunks:
        st.subheader("Context Used:")
        for chunk_id, _ in relevant_chunks:
            st.text(f"Chunk ID: {chunk_id}")

    # Feedback Section
    st.subheader("Was this answer helpful?")
    feedback = st.radio("", ["Yes", "No"], index=None, key="feedback")
    if feedback:
        rag.tools.store_feedback(chunk_id, feedback)
        st.success("Thank you for your feedback!")
