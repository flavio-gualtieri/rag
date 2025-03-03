```
# Chat with Your PDF (RAG System)

## Overview
This project implements a **Retrieval-Augmented Generation (RAG) system** for interacting with PDF documents. Users can upload a PDF, and the system will chunk and store its text in Google BigQuery. It then retrieves the most relevant document chunks based on user queries and generates responses using **Google's Gemini model**.

## Features
- **PDF Processing**: Extracts text from PDFs, chunks it, and stores embeddings in BigQuery.
- **Semantic Search**: Retrieves relevant document chunks using **cosine similarity**.
- **LLM Integration**: Uses **Google Gemini** to generate answers based on retrieved text.
- **Web App Interface**: Provides an **interactive chat** through **Streamlit**.
- **Feedback Collection**: Allows users to rate responses and store feedback in BigQuery.

## Technologies Used
- **Python**
- **Google BigQuery** (for document storage and retrieval)
- **Google Gemini API** (for language generation and embeddings)
- **SentenceTransformers** (for text embeddings)
- **Streamlit** (for user interface)
- **PyPDF2** (for PDF text extraction)
- **pandas, numpy, logging** (for data processing and logging)

## Installation
### Prerequisites
- **Google Cloud SDK**
- **Google Cloud BigQuery enabled**
- Python 3.8+
- Required dependencies (install with pip):
  ```bash
  pip install google-cloud-bigquery google-generativeai streamlit PyPDF2 sentence-transformers pandas numpy langchain
  ```

## Usage
### 1. Setting Up the API Key
Store your **Google Gemini API key** in a `config.txt` file:
```
GEMINI_API_KEY=your_api_key_here
```

### 2. Running the Streamlit App
```bash
streamlit run app.py
```

### 3. Uploading a PDF
- Click on **Upload PDF** in the app.
- The system will process and store the document in BigQuery.

### 4. Asking Questions
- Type a question related to the document in the chat input.
- The system retrieves relevant chunks and generates an answer using the **Gemini model**.

### 5. Providing Feedback
- After receiving an answer, you can rate the response and add comments.
- Feedback is stored in **BigQuery** for future improvements.

## Code Structure
```
├── app.py          # Streamlit Web App
├── rag.py          # Main RAG Logic
├── tools.py        # PDF Processing and BigQuery Integration
├── config.txt      # API Key Storage
├── requirements.txt # Python Dependencies
```

## Future Improvements
- **Support for multiple documents** in a single session.
- **Enhanced search ranking** for better chunk retrieval.
- **Integration with additional LLMs** for response generation.

```
