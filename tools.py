import PyPDF2
import io
import logging
from typing import Tuple, Optional, List
import re
import pandas as pd
import uuid
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from google.cloud import bigquery

class Tools:
    def __init__(self, chunk_size: int, overlap: int, embedding_model: str = "all-MiniLM-L6-v2", 
                 project_id: str = "flaviosrag", dataset_id: str = "document_chunks", table_id: str = "vectorized_chunks"):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.client = bigquery.Client()
        self.embedder = SentenceTransformer(embedding_model)
        self.logger = self.__setup_logger()
        self.table_ref = self.__get_table_ref()


    def __setup_logger(self):
        logger = logging.getLogger("udf_logger")
        return logger

    
    def __get_table_ref(self) -> str:
        return f"{self.project_id}.{self.dataset_id}.{self.table_id}"


    def pdf_reader(self, file_path: str) -> Tuple[Optional[str], Optional[List[int]]]:
        self.logger.info(f"Opening {file_path}.")

        try:
            with open(file_path, "rb") as f:
                buffer = io.BytesIO(f.read())

            reader = PyPDF2.PdfReader(buffer)
        except Exception as e:
            self.logger.error(f"Failed to open {file_path}: {e}.")
            return None, None

        metadata = reader.metadata
        metadata_text = f"{metadata.title or ''} {metadata.author or ''}".strip() if metadata else ""

        text = ""
        page_breaks = [0]

        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    page_text = page_text.replace("\n", " ").replace("\x00", "")

                    if metadata_text and page_text.startswith(metadata_text):
                        page_text = page_text[len(metadata_text):].strip()

                    text += page_text + " "
                    page_breaks.append(len(text))
                else:
                    self.logger.warning(f"Empty text extracted from {file_path}, page {i + 1}.")
            except Exception as e:
                self.logger.warning(f"Unable to extract text from {file_path}, page {i + 1}: {e}")

        if not text.strip() or len(text) < 1500:
            self.logger.info(f"File {file_path} is empty or has insufficient text for extraction.")
            return None, None

        return text, page_breaks


    def text_chunker(self, text: str, document_name: str) -> pd.DataFrame:
        self.logger.info("Chunking text.")
        text = re.sub(r"\s+", " ", text).strip()

        splitter = CharacterTextSplitter(separator=" ", chunk_size=self.chunk_size, chunk_overlap=self.overlap)
        chunks = splitter.split_text(text)

        self.logger.info(f"Text split into {len(chunks)} chunks.")

        # Compute embeddings
        embeddings = self.embedder.encode(chunks).tolist()

        # Create DataFrame
        df = pd.DataFrame({
            "uuid": [str(uuid.uuid4()) for _ in range(len(chunks))],
            "chunk": chunks,
            "embedding": embeddings,
            "document_name": [document_name] * len(chunks)
        })

        return df


    def push_df_to_db(self, df: pd.DataFrame, document_name: str):
        rows = [
            {
                "UUID": row["uuid"],
                "CHUNK": row["chunk"],
                "EMBEDDING": json.dumps(row["embedding"]),  # Store embeddings as JSON
                "DOCUMENT_NAME": row["document_name"]  # New field
            }
            for _, row in df.iterrows()
        ]

        errors = self.client.insert_rows_json(self.table_ref, rows)
        if errors:
            print(f"Failed to insert rows: {errors}")
        else:
            print(f"Successfully inserted {len(rows)} rows into BigQuery.")


""" import os

# Define file path (replace 'path' with your actual PDF file path)
file_path = "/Users/flaviogualtieri/Downloads/test1.pdf"
document_name = os.path.basename(file_path)

# Initialize the Tools object
tools = Tools(chunk_size=500, overlap=100)

# Step 1: Read the PDF
text, page_breaks = tools.pdf_reader(file_path)
if text is None:
    print("Failed to extract text from PDF.")
    exit()

# Step 2: Chunk the text
df_chunks = tools.text_chunker(text, document_name)
if df_chunks.empty:
    print("Failed to chunk text.")
    exit()

# Step 3: Push to BigQuery
tools.push_df_to_db(df_chunks, document_name)
 """