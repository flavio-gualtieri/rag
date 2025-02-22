import logging
from google.cloud import bigquery
import numpy as np
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
from tools import Tools  # Assuming the Tools class handles PDF reading, chunking, and embedding storage
import google.generativeai as genai
import json
import pandas as pd

class RAG:
    def __init__(self, tools: Tools, chunks=20, window=10):
        self.gemini_api_key = self.__get_api_key()
        self.chunks = chunks
        self.window = window
        self.tools = tools
        self.client = bigquery.Client()
        self.logger = self.__init_logger()
        self.model_name = "gemini-1.5-flash"
        
        # Initialize Gemini API
        genai.configure(api_key=self.gemini_api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def __init_logger(self):
        logger = logging.getLogger("RAG_LOGGER")
        return logger

    def __get_api_key(self, filepath="config.txt"):
        with open(filepath, "r") as file:
            for line in file:
                if line.startswith("GEMINI_API_KEY="):
                    return line.strip().split("=")[1]

    def process_pdf(self, file_path: str, document_name: str) -> pd.DataFrame:
        try:
            text, _ = self.tools.pdf_reader(file_path)
            self.logger.info("Extracted text")
        except Exception as e:
            self.logger.warning(f"Failed to obtain text from pdf: {e}")
        if text:
            try:
                chunk_df = self.tools.text_chunker(text)
                self.logger.info("Chunked and embedded")
            except Exception as e:
                self.logger.warning(f"Failed to chunk and embed:{e}")
            try:
                self.tools.push_df_to_db(chunk_df, document_name)
                self.logger.info("Pushed to df")
            except Exception as e:
                self.logger.warning(f"Failed to push to db: {e}")
            return chunk_df
        return None

    def retrieve_relevant_chunks(self, question: str, top_k=5, document_name: str = None):
        question_vector = self.tools.embedder.encode([question])[0]

        # Query BigQuery for stored document chunks
        query = f"""
        SELECT uuid, chunk, vector, document_name
        FROM `{self.tools.get_table_ref()}`
        """
        if document_name:
            query += f" WHERE document_name = '{document_name}'"

        results = self.client.query(query).result()

        embeddings = []
        chunk_texts = []
        chunk_ids = []

        for row in results:
            try:
                vector = np.array(json.loads(row["vector"]), dtype=np.float32)  # Convert JSON string to NumPy array
                embeddings.append(vector)
                chunk_texts.append(row["chunk"])
                chunk_ids.append(row["uuid"])
            except Exception as e:
                print(f"Error processing row {row['uuid']}: {e}")

        if not embeddings:
            return []

        embeddings = np.stack(embeddings)  # Convert list of arrays into a 2D NumPy array

        # Compute cosine similarity
        similarities = cosine_similarity([question_vector], embeddings)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]

        relevant_chunks = [(chunk_ids[i], chunk_texts[i], similarities[i]) for i in top_indices]
        return relevant_chunks



    def generate_answer(self, question: str):
        self.logger.info("Rephrasing question using LLM.")
        rephrased_question = self.rephrase_question(question)
        
        relevant_chunks = self.retrieve_relevant_chunks(rephrased_question)
        
        # Retrieve chunk texts
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        chunk_texts = []
        for chunk_id, _ in relevant_chunks:
            cursor.execute("SELECT vector FROM embeddings WHERE id=?", (chunk_id,))
            result = cursor.fetchone()
            if result:
                chunk_texts.append(result[0].decode("utf-8"))
        conn.close()
        
        prompt = self.__create_prompt(rephrased_question, chunk_texts)
        return self.generate_text(prompt)

    def __create_prompt(self, question, context_chunks):
        context = "\n\n".join(context_chunks)
        return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    def rephrase_question(self, question: str) -> str:
        response = self.model.generate_content(f"Rephrase the following question for clarity: {question}")
        return response.text.strip()

    def generate_text(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text.strip()
