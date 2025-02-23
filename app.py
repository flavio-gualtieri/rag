import streamlit as st
import os
import pandas as pd
from tools import Tools
from rag import RAG
import tempfile

# Initialize Streamlit app
st.title("Chat with Your PDF")

# Cache tools and RAG initialization
@st.cache_resource
def get_rag():
    tools = Tools(chunk_size=800, overlap=100)
    return RAG(tools)

rag = get_rag()

col1, col2 = st.columns([3, 1])

with col1:
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_path = tmp_file.name

        document_name = os.path.splitext(os.path.basename(file_path))[0]

        if "document_name" not in st.session_state:
            st.write("Processing PDF...")
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
    if uploaded_file:
        user_input = st.chat_input("Ask a question about the document...")

        if user_input:
            st.session_state["messages"].append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            # Generate response
            @st.cache_data
            def generate_answer(question, document_name):
                return rag.generate_answer(question, document_name)

            with st.spinner("Retrieving relevant information..."):
                answer = generate_answer(user_input, st.session_state.get("document_name", ""))

            with st.chat_message("assistant"):
                st.write(answer)

            st.session_state["messages"].append({"role": "assistant", "content": answer})
            st.session_state["last_answer"] = answer  
            st.session_state["last_question"] = user_input

if "last_answer" in st.session_state:
    with col2:
        with st.container():
            st.markdown("Feedback Form", unsafe_allow_html=True)
            rating = st.feedback("stars")
            st.session_state["rating"] = rating
            notes = st.text_area("Additional comments")

            if st.button("Submit Feedback"):
                feedback_entry = {
                    "question": st.session_state.get("last_question", ""),
                    "answer": st.session_state["last_answer"],
                    "rating": rating,
                    "notes": notes,
                    "document_name": st.session_state.get("document_name", ""),
                }

                if "feedback_df" not in st.session_state:
                    st.session_state["feedback_df"] = pd.DataFrame(columns=["question", "answer", "rating", "notes", "document_name"])

                st.session_state["feedback_df"] = pd.concat(
                    [st.session_state["feedback_df"], pd.DataFrame([feedback_entry])], ignore_index=True
                )

                rag.tools.push_feedback(st.session_state["feedback_df"])

                st.success("Thank you for your feedback! It has been recorded.")
