import PyPDF2
import io
import logging
from typing import Tuple, Optional, List
import re
import pandas as pd
import uuid
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import sqlite3
import numpy as np

class Tools:
    def __init__(self, chunk_size: int, overlap: int, embedding_model: str = "all-MiniLM-L6-v2", db_path: str = "embeddings.db"):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.db_path = db_path
        self.logger = self.__setup_logger()
        self.embedder = SentenceTransformer(embedding_model)
        self.__setup_db()


    def __setup_logger(self):
        logger = logging.getLogger("udf_logger")
        return logger


    def __setup_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id TEXT PRIMARY KEY,
            vector BLOB)
        """)
        conn.commit()
        conn.close()


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

    def text_chunker(self, text: str) -> pd.DataFrame:
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
            "embedding": embeddings
        })

        return df

    def push_df_to_db(self, df: pd.DataFrame):
        """Stores a DataFrame of text chunks and embeddings in SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for _, row in df.iterrows():
            # Convert embedding array to bytes
            vector_bytes = np.array(row["embedding"], dtype=np.float32).tobytes()
            cursor.execute("INSERT INTO embeddings (id, vector) VALUES (?, ?)", (row["uuid"], vector_bytes))
        
        conn.commit()
        conn.close()



# Example usage
tools = Tools(chunk_size=800, overlap=100)

pdf_text, _ = tools.pdf_reader("/Users/flaviogualtieri/Desktop/example/introduciton_to_python-1.pdf")

if pdf_text:
    chunk_df = tools.text_chunker(pdf_text)
    chunk_df.to_excel("/Users/flaviogualtieri/Desktop/example/output.xlsx", index=False)



